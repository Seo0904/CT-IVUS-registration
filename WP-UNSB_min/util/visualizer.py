import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

from .wandb_logger import WandbLogger

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        # シーケンス (B,T,C,H,W) の場合は、フレームごとに保存
        if hasattr(im_data, 'dim') and callable(im_data.dim) and im_data.dim() == 5:
            # テストスクリプトでは batch_size=1 なので、B=1 前提
            b, t, c, h, w = im_data.shape
            os.makedirs(os.path.join(image_dir, label), exist_ok=True)

            for frame_idx in range(t):
                # (1,C,H,W) になるようにスライスして tensor2im に渡す
                frame_tensor = im_data[0, frame_idx:frame_idx + 1]
                im = util.tensor2im(frame_tensor)
                image_name = '%s/%s_frame_%d.png' % (label, name, frame_idx)
                save_path = os.path.join(image_dir, image_name)
                util.save_image(im, save_path, aspect_ratio=aspect_ratio)
                ims.append(image_name)
                txts.append(f"{label}_frame_{frame_idx}")
                links.append(image_name)
        else:
            # 通常の画像 (B,C,H,W) や (C,H,W) の場合は従来通り 1枚だけ保存
            im = util.tensor2im(im_data)
            image_name = '%s/%s.png' % (label, name)
            os.makedirs(os.path.join(image_dir, label), exist_ok=True)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=aspect_ratio)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.wandb = WandbLogger(opt)
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10  # just a random display id
        else:
            self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            else:
                self.vis = visdom.Visdom(port=2004,
                                         base_url=os.environ['tensorboard_base_url'] + '/visdom')
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, global_step=None, split='train'):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, ncols, 2, self.display_id + 1,
                                    None, dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(
                            image_numpy.transpose([2, 0, 1]),
                            self.display_id + idx,
                            None,
                            dict(title=label)
                        )
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            # Optional W&B image logging
            self._wandb_log_visuals(visuals, epoch, global_step=global_step, split=split)

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def _tensor_to_wandb_image(self, tensor, caption=None):
        """Convert a tensor (B,C,H,W) or (B,T,C,H,W) into a stitched RGB uint8 image for wandb."""
        if not self.wandb.enabled or self.wandb.wandb is None:
            return None

        try:
            import numpy as _np
            import torch

            if isinstance(tensor, _np.ndarray):
                img = tensor
                if img.ndim == 2:
                    img = _np.tile(img[..., None], (1, 1, 3))
                return self.wandb.wandb.Image(img, caption=caption)

            if not isinstance(tensor, torch.Tensor):
                return None

            x = tensor.detach().cpu()

            # sequence: (B,T,C,H,W) -> stitch frames horizontally for the first sample
            if x.dim() == 5:
                x = x[0]  # (T,C,H,W)
                t, c, h, w = x.shape
                x = x.clamp(-1.0, 1.0)
                x = (x + 1.0) / 2.0
                x = _np.clip(x.numpy(), 0.0, 1.0)
                if c == 1:
                    x = _np.tile(x, (1, 3, 1, 1))
                    c = 3
                canvas = _np.zeros((h, w * t, c), dtype=_np.float32)
                for i in range(t):
                    frame = x[i].transpose(1, 2, 0)  # (H,W,C)
                    canvas[:, i * w:(i + 1) * w, :] = frame
                img = (canvas * 255.0).astype(_np.uint8)
                return self.wandb.wandb.Image(img, caption=caption)

            # image: (B,C,H,W) or (C,H,W)
            if x.dim() == 4:
                x = x[0]
            if x.dim() != 3:
                return None
            x = x.clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0
            x = x.numpy()
            if x.shape[0] == 1:
                x = _np.tile(x, (3, 1, 1))
            img = (x.transpose(1, 2, 0) * 255.0).astype(_np.uint8)
            return self.wandb.wandb.Image(img, caption=caption)
        except Exception as e:
            print(f"[wandb] image conversion failed: {e}")
            return None

    def _wandb_log_visuals(self, visuals, epoch, global_step=None, split='train'):
        if not self.wandb.enabled:
            return
        if not getattr(self.opt, 'wandb_log_images', True):
            return

        freq = int(getattr(self.opt, 'wandb_image_freq', 2000) or 2000)
        if global_step is not None and freq > 0 and (global_step % freq) != 0:
            return

        payload = {'epoch': int(epoch)}
        for label, tensor in visuals.items():
            wimg = self._tensor_to_wandb_image(tensor, caption=f"{split}:{label} (epoch={epoch})")
            if wimg is not None:
                payload[f"{split}/visuals/{label}"] = wimg

        if payload:
            self.wandb.log(payload, step=global_step)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
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
                opts={
                    'title': self.name,
                    'legend': plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id - plot_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, global_step=None, split='train'):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        # W&B scalar logging
        if self.wandb.enabled:
            payload = {
                'epoch': float(epoch),
                f"{split}/time": float(t_comp),
            }
            if split == 'val':
                payload['val/epoch'] = float(epoch)
            for k, v in losses.items():
                try:
                    wandb_key = k
                    if split == 'val' and isinstance(k, str) and k.startswith('val_'):
                        wandb_key = k[4:]
                    payload[f"{split}/{wandb_key}"] = float(v)
                except Exception:
                    pass
            self.wandb.log(payload, step=global_step)

    def wandb_log_visuals(self, visuals, epoch, global_step=None, split='train'):
        """Public wrapper for logging visuals to W&B with the same frequency control."""
        self._wandb_log_visuals(visuals, epoch, global_step=global_step, split=split)

    def wandb_log_files_as_images(self, image_paths, global_step=None, split='train', prefix='debug', epoch=None):
        """Log local PNG/JPG files as W&B images."""
        if not self.wandb.enabled or self.wandb.wandb is None:
            return
        payload = {}
        if epoch is not None:
            try:
                payload['epoch'] = int(epoch)
            except Exception:
                pass
        for p in image_paths:
            try:
                payload[f"{split}/{prefix}/{os.path.basename(p)}"] = self.wandb.wandb.Image(p)
            except Exception:
                continue
        if payload:
            self.wandb.log(payload, step=global_step)
