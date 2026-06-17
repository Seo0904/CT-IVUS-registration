"""
同じ val シーケンスについて、指定した複数 epoch の生成結果を縦に並べて比較する。

行構成（1シーケンスあたり）:
    real_A      : 入力（穴なし）
    real_B(GT)  : 目標（穴なし版を渡す）
    fake@ep0    : epoch ep0 の生成
    fake@ep1    : epoch ep1 の生成
    ...
列 = フレーム（時系列）

各フレームは [-1,1] -> [0,255] の固定スケールで描画する（FID 計算の
per-image normalize とは違い、epoch 間で明るさが比較可能）。

使い方は run_visualize_epoch_compare.sh を参照。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def _pop_arg(argv, key, default):
    if key in argv:
        i = argv.index(key)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def to_gray(frame: torch.Tensor) -> np.ndarray:
    """(C,H,W) in [-1,1] -> (H,W) uint8。固定スケール。"""
    x = frame.detach().cpu().float()
    if x.dim() == 3:
        x = x[0]  # 1ch 前提で先頭チャネル
    x = x.clamp(-1, 1)
    return (((x + 1) / 2) * 255).round().byte().numpy()


def strip(frames, pad=2):
    h, w = frames[0].shape
    out = np.zeros((h, len(frames) * w + (len(frames) - 1) * pad), np.uint8)
    for i, f in enumerate(frames):
        out[:, i * (w + pad): i * (w + pad) + w] = f
    return out


def label_col(h, text, w=90):
    img = Image.fromarray(np.zeros((h, w), np.uint8))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    bb = d.textbbox((0, 0), text, font=font)
    d.text(((w - (bb[2] - bb[0])) // 2, (h - (bb[3] - bb[1])) // 2), text, fill=230, font=font)
    return np.array(img)


def main():
    epochs = [int(e) for e in _pop_arg(sys.argv, '--epochs', '320,710').split(',')]
    num_seq = int(_pop_arg(sys.argv, '--num_seq', '5'))
    out_dir = _pop_arg(sys.argv, '--out_dir', '/tmp/epoch_compare')

    opt = TestOptions().parse()
    opt.phase = 'val'
    opt.isTrain = False
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    ds = create_dataset(opt)
    ds2 = create_dataset(opt)

    # 先頭 num_seq シーケンスを取り出す
    samples = []
    it2 = iter(ds2)
    for i, data in enumerate(ds):
        data2 = next(it2)
        if i == 0:
            first_data, first_data2 = data, data2
        samples.append((data, data2))
        if len(samples) >= num_seq:
            break

    model = create_model(opt)
    model.data_dependent_initialize(first_data, first_data2)
    model.setup(opt)
    model.eval()

    # 各サンプルの real_A / real_B と、epoch ごとの fake を収集
    # fakes[seq_idx][epoch] = list[(H,W)]
    real_A = {}
    real_B = {}
    fakes = {s: {} for s in range(len(samples))}

    with torch.no_grad():
        for ep in epochs:
            model.load_networks(str(ep))
            for s, (data, data2) in enumerate(samples):
                model.set_input(data, data2)
                model.forward()
                fb = model.fake_B  # (1,T,C,H,W) or (1,C,H,W)
                rb = model.real_B
                ra = model.real_A
                fb = fb[0] if fb.dim() == 5 else fb
                rb = rb[0] if rb.dim() == 5 else rb
                ra = ra[0] if ra.dim() == 5 else ra
                fakes[s][ep] = [to_gray(fb[t]) for t in range(fb.shape[0])]
                if s not in real_A:
                    real_A[s] = [to_gray(ra[t]) for t in range(ra.shape[0])]
                    real_B[s] = [to_gray(rb[t]) for t in range(rb.shape[0])]

    os.makedirs(out_dir, exist_ok=True)
    for s in range(len(samples)):
        rows = []
        labels = []
        rows.append(strip(real_A[s])); labels.append('real_A')
        rows.append(strip(real_B[s])); labels.append('real_B(GT)')
        for ep in epochs:
            rows.append(strip(fakes[s][ep])); labels.append(f'fake@{ep}')

        h_each, total_w = rows[0].shape
        canvas = np.zeros((len(rows) * (h_each + 2), 90 + total_w), np.uint8)
        y = 0
        for row, lab in zip(rows, labels):
            canvas[y:y + h_each, :90] = label_col(h_each, lab)
            canvas[y:y + h_each, 90:] = row
            y += h_each + 2
        out_path = os.path.join(out_dir, f'compare_seq{s}.png')
        Image.fromarray(canvas).save(out_path)
        print(f'saved: {out_path}')

    print(f'\nDone. {len(samples)} sequences -> {out_dir}')
    print(f'epochs compared: {epochs}')


if __name__ == '__main__':
    main()
