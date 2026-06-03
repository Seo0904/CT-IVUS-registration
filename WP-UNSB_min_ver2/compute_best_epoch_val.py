"""
保存済みチェックポイントに対して、val を「穴なし(non-holed) GT」で評価し直し、
FID 最小の best epoch を計算するスクリプト。

train.py の run_fid をそのまま再利用するので、学習時と同じ方法で FID を測る。
唯一の違いは GT に穴なし版 (transformed_global.npy) を使う点。
（穴なし版は get_valid_frame_idx で除外されるゼロフレームが無いので、全フレームが評価対象になる）

使い方:
    python compute_best_epoch_val.py \
        --dataroot   .../moving_mnist \
        --dataroot_B .../bspline_transformed \
        --data_file_A mnist_test_seq.npy \
        --data_file_B transformed_global.npy   # ← 穴なし版
        --name moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze \
        --checkpoints_dir .../<DATE> \
        --epoch_start 10 --epoch_end 800 --epoch_step 10 \
        ...（train/test と同じモデル系オプション）
"""
import os
import sys
import json

import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# 学習時とまったく同じ FID 計算を使う
from train import run_fid


def _pop_arg(argv, key, default):
    """sys.argv から '--key value' を取り除いて値を返す（TestOptions に渡さないため）。"""
    if key in argv:
        i = argv.index(key)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def main():
    # TestOptions が知らない独自オプションを先に抜き取る
    epoch_start = int(_pop_arg(sys.argv, '--epoch_start', 10))
    epoch_end = int(_pop_arg(sys.argv, '--epoch_end', 800))
    epoch_step = int(_pop_arg(sys.argv, '--epoch_step', 10))

    opt = TestOptions().parse()

    # val 固定設定
    opt.phase = 'val'
    opt.isTrain = False
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # val データセット（穴なし GT）
    val_dataset = create_dataset(opt)
    val_dataset2 = create_dataset(opt)

    # モデルを 1 度だけ構築。data_dependent_initialize → setup（ここで opt.epoch を一旦ロード）
    model = create_model(opt)
    first_data = first_data2 = None
    for d, d2 in zip(val_dataset, val_dataset2):
        first_data, first_data2 = d, d2
        break
    model.data_dependent_initialize(first_data, first_data2)
    model.setup(opt)        # opt.epoch のチェックポイントをロード（後で上書きする）
    model.eval()

    results = {}
    candidate_epochs = list(range(epoch_start, epoch_end + 1, epoch_step))

    for epoch in candidate_epochs:
        # 該当エポックの重みが存在するか確認
        ckpt = os.path.join(opt.checkpoints_dir, opt.name, f'{epoch}_net_G.pth')
        if not os.path.exists(ckpt):
            print(f'[skip] epoch {epoch}: checkpoint not found ({ckpt})')
            continue

        model.load_networks(str(epoch))
        fid = run_fid(model, opt, epoch, val_dataset)
        results[epoch] = float(fid)

    if not results:
        print('No epochs evaluated. Check checkpoint paths.')
        return

    # best（FID 最小）
    best_epoch = min(results, key=results.get)
    best_fid = results[best_epoch]

    print('\n' + '=' * 60)
    print('Best-epoch search on val with NON-HOLED GT')
    print(f'  Domain B (val GT): {opt.data_file_B}')
    print(f'  Epoch range: {epoch_start}-{epoch_end} step {epoch_step}')
    print('=' * 60)
    for ep in sorted(results):
        mark = '  <== BEST' if ep == best_epoch else ''
        print(f'  epoch {ep:>4}: FID = {results[ep]:.4f}{mark}')
    print('=' * 60)
    print(f'BEST epoch = {best_epoch}  (FID = {best_fid:.4f})')
    print('=' * 60)

    # 保存
    out = {
        'data_file_B': opt.data_file_B,
        'epoch_range': [epoch_start, epoch_end, epoch_step],
        'best_epoch': best_epoch,
        'best_fid': best_fid,
        'fid_per_epoch': results,
    }
    tag = os.path.splitext(os.path.basename(opt.data_file_B))[0]
    out_path = os.path.join(opt.checkpoints_dir, f'best_epoch_val_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
