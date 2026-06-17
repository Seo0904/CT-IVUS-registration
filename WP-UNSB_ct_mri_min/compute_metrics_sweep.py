"""
保存済みチェックポイントに対して、val を「穴なし(non-holed) GT」とのペア指標
(SSIM / PSNR / L2 / L1) で全 epoch 評価し、知覚的に best な epoch を探す。

FID は Inception(ImageNet) ベースで Moving MNIST には不向きなため、
ピクセルレベルのペア指標で「見た目の良さ」に近い順位を出す目的。

使い方は run_compute_metrics_sweep.sh を参照。GT は穴なし版を渡すこと
（穴なしなら get_valid_frame_idx が全フレームを残すので、全 20 フレームのペア評価になる）。
"""
import os
import sys
import json

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# 学習時とまったく同じ paired metrics 計算を使う
from train import run_validation


def _pop_arg(argv, key, default):
    if key in argv:
        i = argv.index(key)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def main():
    epoch_start = int(_pop_arg(sys.argv, '--epoch_start', 10))
    epoch_end = int(_pop_arg(sys.argv, '--epoch_end', 800))
    epoch_step = int(_pop_arg(sys.argv, '--epoch_step', 10))

    opt = TestOptions().parse()
    opt.phase = 'val'
    opt.isTrain = False
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    # 全 val シーケンスを使う
    opt.num_val_samples = 100000

    val_dataset = create_dataset(opt)
    val_dataset2 = create_dataset(opt)

    model = create_model(opt)
    first_data = first_data2 = None
    for d, d2 in zip(val_dataset, val_dataset2):
        first_data, first_data2 = d, d2
        break
    model.data_dependent_initialize(first_data, first_data2)
    model.setup(opt)
    model.eval()

    results = {}
    for epoch in range(epoch_start, epoch_end + 1, epoch_step):
        ckpt = os.path.join(opt.checkpoints_dir, opt.name, f'{epoch}_net_G.pth')
        if not os.path.exists(ckpt):
            print(f'[skip] epoch {epoch}: checkpoint not found')
            continue
        model.load_networks(str(epoch))
        m = run_validation(model, opt, epoch, val_dataset)
        results[epoch] = {k: float(v) for k, v in m.items()}

    if not results:
        print('No epochs evaluated.')
        return

    best_ssim_ep = max(results, key=lambda e: results[e]['SSIM'])
    best_psnr_ep = max(results, key=lambda e: results[e]['PSNR'])
    best_l2_ep = min(results, key=lambda e: results[e]['L2'])

    print('\n' + '=' * 70)
    print('Paired metrics sweep on val with NON-HOLED GT (all 20 frames)')
    print(f'  Domain B (val GT): {opt.data_file_B}')
    print('=' * 70)
    print(f'{"epoch":>6} | {"SSIM":>8} | {"PSNR":>8} | {"L2":>10} | {"L1":>10}')
    print('-' * 70)
    for ep in sorted(results):
        r = results[ep]
        marks = []
        if ep == best_ssim_ep:
            marks.append('SSIM*')
        if ep == best_psnr_ep:
            marks.append('PSNR*')
        if ep == best_l2_ep:
            marks.append('L2*')
        mark = ('  <== ' + ','.join(marks)) if marks else ''
        print(f'{ep:>6} | {r["SSIM"]:>8.4f} | {r["PSNR"]:>8.4f} | {r["L2"]:>10.5f} | {r["L1"]:>10.5f}{mark}')
    print('=' * 70)
    print(f'BEST by SSIM : epoch {best_ssim_ep}  (SSIM = {results[best_ssim_ep]["SSIM"]:.4f})')
    print(f'BEST by PSNR : epoch {best_psnr_ep}  (PSNR = {results[best_psnr_ep]["PSNR"]:.4f})')
    print(f'BEST by L2   : epoch {best_l2_ep}  (L2 = {results[best_l2_ep]["L2"]:.5f})')
    print('=' * 70)

    out = {
        'data_file_B': opt.data_file_B,
        'epoch_range': [epoch_start, epoch_end, epoch_step],
        'best_by_ssim': best_ssim_ep,
        'best_by_psnr': best_psnr_ep,
        'best_by_l2': best_l2_ep,
        'metrics_per_epoch': results,
    }
    out_path = os.path.join(opt.checkpoints_dir, 'metrics_sweep_val.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
