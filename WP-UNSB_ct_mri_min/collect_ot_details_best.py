#!/usr/bin/env python3
"""
best epoch で学習済みモデルをロードし，train_data から num_samples シーケンス分の
OT detail (P, M, U, 画像) を ot_details/train_best/ に保存するスクリプト。

Usage:
    python collect_ot_details_best.py \
        --checkpoints_dir /workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919 \
        --name moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze \
        --num_samples 10
"""
import os
import sys
import argparse
import torch

# WP-UNSB_min_ver2 ディレクトリを sys.path に追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.sequence_ot import sequence_ot_loss_torch


def build_train_snapshot_opt(opt):
    import argparse as _ap
    tr_opt = _ap.Namespace(**{k: v for k, v in vars(opt).items() if k != 'visualizer'})
    tr_opt.phase = 'train'
    tr_opt.isTrain = False
    tr_opt.serial_batches = True
    tr_opt.no_flip = True
    tr_opt.batch_size = 1
    return tr_opt


def main():
    parser = argparse.ArgumentParser(description='Collect OT details from best epoch model on train data.')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='/workspace/data/experiment_result/WP-UNSB_min_ver2/moving-mnist/20260527_125919')
    parser.add_argument('--name', type=str,
                        default='moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of train sequences to collect OT details for')
    parser.add_argument('--out_subdir', type=str, default='train_best',
                        help='Subdirectory name under ot_details/ to save results')
    args = parser.parse_args()

    # train_opt.txt から学習設定を再現する
    # TrainOptions を直接 parse せず、固定引数で組み立てる
    opt_args = [
        '--checkpoints_dir', args.checkpoints_dir,
        '--name', args.name,
        '--dataroot', '/workspace/data/org_data/moving_mnist',
        '--dataroot_B', '/workspace/data/preprocessed/bspline_transformed',
        '--data_file_A', 'mnist_test_seq.npy',
        '--data_file_B', 'transformed_cloze_global.npy',
        '--model', 'wpsb',
        '--mode', 'sb',
        '--dataset_mode', 'moving_mnist_paired',
        '--input_nc', '1',
        '--output_nc', '1',
        '--ngf', '64',
        '--ndf', '64',
        '--load_size', '64',
        '--crop_size', '64',
        '--batch_size', '4',          # TrainOptions 用（snapshot では 1 に上書きする）
        '--num_timesteps', '5',
        '--lambda_GAN', '0.0',
        '--lambda_SB', '1.0',
        '--lambda_NCE', '0.0',
        '--lmda', '0.10',
        '--seq_ot_iters', '150',
        '--sinkhorn_type', 'sinkhorn_log',
        '--sb_mode', 'seq_ot',
        '--seq_ot_normalize', 'none',
        '--seq_ot_p_entropy', '0',
        '--seq_ot_p_entropy_penalty', '0.0',
        '--seq_ot_divergence', '1',
        '--seq_ot_divergence_penalty', '-0.5',
        '--seq_ot_monotone_penalty', '1.0',
        '--n_epochs', '800',
        '--n_epochs_decay', '0',
        '--lr', '0.0002',
        '--save_epoch_freq', '10',
        '--preprocess', 'none',
        '--no_flip',
        '--display_id', '0',
        '--num_threads', '0',
        '--gpu_ids', '0',
        '--save_ot_details',
        '--ot_details_max_samples', str(args.num_samples),
        '--epoch', 'best',           # best epoch をロード
        '--no_html',
    ]

    sys.argv = [sys.argv[0]] + opt_args
    opt = TrainOptions().parse()
    # isTrain=False にして load_networks (best) を走らせる
    opt.isTrain = False

    # snapshot 用オプション（train, serial, batch=1）
    snap_opt = build_train_snapshot_opt(opt)

    # モデル構築 & best epoch ロード
    print(f"[Info] Loading model (epoch=best) from {args.checkpoints_dir}/{args.name}")
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # OT 詳細の出力ディレクトリ
    ot_base = os.path.join(args.checkpoints_dir, args.name, 'ot_details')
    out_dir = os.path.join(ot_base, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    model.ot_details_dir = out_dir
    model.ot_details_saved = 0

    print(f"[Info] Output dir: {out_dir}")
    print(f"[Info] Collecting OT details for {args.num_samples} train sequences ...")

    train_dataset = create_dataset(snap_opt)

    with torch.no_grad():
        for i, data in enumerate(train_dataset):
            if i >= args.num_samples:
                break

            model.set_input(data)
            model.forward()

            fake_seq = model.fake_B
            real_seq = model.real_B

            if fake_seq.dim() == 5 and real_seq.dim() == 5:
                b = fake_seq.shape[0]
                for bi in range(min(b, 1)):
                    ot_val, details = sequence_ot_loss_torch(
                        fake_seq[bi], real_seq[bi],
                        reg=opt.lmda,
                        iters=getattr(opt, 'seq_ot_iters', 50),
                        monotone=getattr(opt, 'seq_ot_monotone', True),
                        monotone_penalty=getattr(opt, 'seq_ot_monotone_penalty', 50.0),
                        P_entropy=getattr(opt, 'seq_ot_p_entropy', False),
                        P_entropy_penalty=getattr(opt, 'seq_ot_p_entropy_penalty', 0.0),
                        ot_divergence=getattr(opt, 'seq_ot_divergence', False),
                        ot_divergence_penalty=getattr(opt, 'seq_ot_divergence_penalty', -0.5),
                        normalize=None,  # seq_ot_normalize == 'none'
                        sinkhorn_type=getattr(opt, 'sinkhorn_type', 'sinkhorn'),
                        return_details=True,
                    )
                    if details is not None:
                        real_A_seq = None
                        try:
                            real_A = getattr(model, 'real_A', None)
                            if real_A is not None:
                                real_A_seq = real_A[bi] if real_A.dim() == 5 else real_A
                        except Exception:
                            pass
                        model._save_ot_details(
                            details,
                            real_A_seq=real_A_seq,
                            real_B_seq=real_seq[bi],
                            fake_B_seq=fake_seq[bi],
                        )
                        saved_idx = model.ot_details_saved
                        model.ot_details_saved += 1
                        print(f"  [{i+1}/{args.num_samples}] saved ot_{saved_idx:04d}  ot_val={ot_val.item():.4f}")
                    else:
                        print(f"  [{i+1}/{args.num_samples}] details is None (skipped)")
            else:
                ot_val, details = sequence_ot_loss_torch(
                    fake_seq, real_seq,
                    reg=opt.lmda,
                    iters=getattr(opt, 'seq_ot_iters', 50),
                    monotone=getattr(opt, 'seq_ot_monotone', True),
                    monotone_penalty=getattr(opt, 'seq_ot_monotone_penalty', 50.0),
                    P_entropy=getattr(opt, 'seq_ot_p_entropy', False),
                    P_entropy_penalty=getattr(opt, 'seq_ot_p_entropy_penalty', 0.0),
                    ot_divergence=getattr(opt, 'seq_ot_divergence', False),
                    ot_divergence_penalty=getattr(opt, 'seq_ot_divergence_penalty', -0.5),
                    normalize=None,
                    sinkhorn_type=getattr(opt, 'sinkhorn_type', 'sinkhorn'),
                    return_details=True,
                )
                if details is not None:
                    real_A_seq = getattr(model, 'real_A', None)
                    model._save_ot_details(
                        details,
                        real_A_seq=real_A_seq,
                        real_B_seq=real_seq,
                        fake_B_seq=fake_seq,
                    )
                    saved_idx = model.ot_details_saved
                    model.ot_details_saved += 1
                    print(f"  [{i+1}/{args.num_samples}] saved ot_{saved_idx:04d}  ot_val={ot_val.item():.4f}")
                else:
                    print(f"  [{i+1}/{args.num_samples}] details is None (skipped)")

    print(f"\n[Done] Saved {model.ot_details_saved} OT detail files to: {out_dir}")


if __name__ == '__main__':
    main()
