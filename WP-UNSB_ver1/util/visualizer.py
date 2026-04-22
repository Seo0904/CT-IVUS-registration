import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import torch

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# ---------------------------
# helpers
# ---------------------------

def _ensure_3ch_chw(x: torch.Tensor) -> torch.Tensor:
    """
    入力: (C,H,W) または (H,W)
    出力: (3,H,W)  ※visdomや保存時の事故防止
    """
    if x.dim() == 2:  # (H,W)
        x = x.unsqueeze(0)   # (1,H,W)

    if x.dim() != 3:
        x = x.squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, f"unexpected dim after squeeze: {x.shape}"

    C, H, W = x.shape
    if C == 1:
        x = x.repeat(3, 1, 1)  # (3,H,W)
    elif C >= 3:
        x = x[:3]  # 念のため先頭3ch
    else:
        # C==2 など変則は 3ch に拡張
        pad = x[:1].repeat(3 - C, 1, 1)
        x = torch.cat([x, pad], dim=0)
    return x


def _pick_three_frames(x, b=None, T=None):
    """
    入力:
      - (B,T,C,H,W)
      - (T,C,H,W)
      - (B*T,C,H,W)
      - (B,C,H,W)
      - (C,H,W)
      - (H,W)
    出力:
      - (3,H,3W)  ※3枚を横に連結（必ず3ch）
    """
    if not torch.is_tensor(x):
        return x

    # (C,H,W) だけ来た場合は3枚に複製
    if x.dim() == 2:
        x = _ensure_3ch_chw(x)                 # (3,H,W)
        return torch.cat([x, x, x], dim=2)     # (3,H,3W)

    if x.dim() == 3:
        x = _ensure_3ch_chw(x)                 # (3,H,W)
        return torch.cat([x, x, x], dim=2)     # (3,H,3W)

    # まず (T,C,H,W) に揃える
    if x.dim() == 5:  # (B,T,C,H,W)
        B, TT = x.size(0), x.size(1)
        if b is None:
            b = np.random.randint(B)
        seq = x[b]  # (T,C,H,W)
        TT = seq.size(0)

    elif x.dim() == 4:
        # (T,C,H,W) or (B*T,C,H,W) or (B,C,H,W)
        N = x.size(0)

        # (B*T,C,H,W) を想定して T,b が分かるなら切る
        if (T is not None) and (b is not None) and (N >= (b + 1) * T):
            seq = x[b * T:(b + 1) * T]  # (T,C,H,W)
            TT = seq.size(0)
        else:
            # (T,C,H,W) として扱う（この場合、TはN）
            seq = x
            TT = seq.size(0)

    else:
        # それ以外は無理やり (C,H,W) にして複製
        x = x.squeeze()
        if x.dim() == 2:
            x = _ensure_3ch_chw(x)
            return torch.cat([x, x, x], dim=2)
        if x.dim() == 3:
            x = _ensure_3ch_chw(x)
            return torch.cat([x, x, x], dim=2)
        raise ValueError(f"_pick_three_frames: unexpected shape {tuple(x.shape)}")

    # 3フレーム選択：先頭/中央/末尾
    idxs = [0, TT // 2, TT - 1]
    frames = []
    for i in idxs:
        fr = seq[i]                 # (C,H,W)
        fr = _ensure_3ch_chw(fr)    # (3,H,W)
        frames.append(fr)

    return torch.cat(frames, dim=2)  # (3,H,3W)


def _make_visuals_np(visuals, b=None, T=None):
    """
    visuals(dict of tensors) -> visuals_np(dict of np.ndarray HWC uint8)
    ここで detach/cpu/tensor2im を 1回だけやる
    """
    visuals_np = {}
    for label, x in visuals.items():
        if torch.is_tensor(x):
            img = _pick_three_frames(x, b=b, T=T)     # (3,H,3W)
            img = img.detach().cpu()
        else:
            img = x
        visuals_np[label] = util.tensor2im(img)        # -> (H,W,3) uint8 (想定)
    return visuals_np


# ---------------------------
# original save_images (used by HTML class sometimes)
# ---------------------------

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    # b,T を決める（同一シーケンスを揃える）
    b = None
    T = None
    for v in visuals.values():
        if torch.is_tensor(v) and v.dim() == 5:
            B, T = v.size(0), v.size(1)
            b = np.random.randint(B)
            break

    visuals_np = _make_visuals_np(visuals, b=b, T=T)

    for label, im_np in visuals_np.items():
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im_np, save_path, aspect_ratio=aspect_ratio)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)

    webpage.add_images(ims, txts, links, width=width)


# ---------------------------
# Visualizer
# ---------------------------

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10
        else:
            self.display_id = opt.display_id

        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False

        if self.display_id > 0:
            import visdom
            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            else:
                self.vis = visdom.Visdom(port=2004, base_url=os.environ['tensorboard_base_url'] + '/visdom')
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        # b,T を決める（全ラベルで同一シーケンス）
        b = None
        T = None
        for v in visuals.values():
            if torch.is_tensor(v) and v.dim() == 5:
                B, T = v.size(0), v.size(1)
                b = np.random.randint(B)
                break

        # ★ここで1回だけnumpy化して使い回す
        visuals_np = _make_visuals_np(visuals, b=b, T=T)

        # --- visdom表示（PNG保存ではない。ブラウザに出すだけ） ---
        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals_np))
                first = next(iter(visuals_np.values()))
                h, w = first.shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)

                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0

                for label, im_np in visuals_np.items():
                    label_html_row += '<td>%s</td>' % label
                    # HWC -> CHW
                    if im_np.ndim == 2:
                        im_np = np.stack([im_np, im_np, im_np], axis=2)
                    images.append(im_np.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''

                white_image = np.ones_like(images[0]) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row

                try:
                    self.vis.images(images, ncols, 2, self.display_id + 1, None, dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:
                idx = 1
                try:
                    for label, im_np in visuals_np.items():
                        if im_np.ndim == 2:
                            im_np = np.stack([im_np, im_np, im_np], axis=2)
                        self.vis.image(im_np.transpose([2, 0, 1]),
                                       self.display_id + idx, None, dict(title=label))
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        # --- HTML/PNG保存 ---
        if self.use_html and (save_result or not self.saved):
            self.saved = True

            for label, im_np in visuals_np.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(im_np, img_path)

            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []
                for label in visuals_np.keys():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        if len(losses) == 0:
            return

        plot_name = '_'.join(list(losses.keys()))
        if plot_name not in self.plot_data:
            self.plot_data[plot_name] = {'X': [], 'Y': [], 'legend': list(losses.keys())}

        plot_data = self.plot_data[plot_name]
        plot_id = list(self.plot_data.keys()).index(plot_name)

        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([losses[k] for k in plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={'title': self.name, 'legend': plot_data['legend'], 'xlabel': 'epoch', 'ylabel': 'loss'},
                win=self.display_id - plot_id
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)