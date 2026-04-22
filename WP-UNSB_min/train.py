# =========================
# 4) train.py（ver2 用）
#    - 学習時は ver1 同様に dataset, dataset2 の 2 本を zip して
#      model.set_input(data, data2) / data_dependent_initialize(data, data2)
#    - validation と FID は必ず val_opt (phase='val') を使う
#    - best は FID最小
# =========================
import time
import os
import torch
import argparse
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from util.metrics import MetricsCalculator
from models.sequence_ot import get_valid_frame_idx, sequence_ot_loss_torch
import shutil
import tempfile
from vgg_sb.evaluations.fid_score import calculate_fid_given_paths
import sys

def build_val_opt(opt):
    val_opt = argparse.Namespace(**{k: v for k, v in vars(opt).items() if k != 'visualizer'})
    val_opt.phase = 'val'
    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.batch_size = 1
    return val_opt


def build_train_snapshot_opt(opt):
    """学習用データから OT 可視化用に同じシーケンスを毎回取るための簡易オプション。

    - phase は 'train' のまま
    - isTrain=False にして、データローダ側のランダム処理を抑える
    - serial_batches=True, no_flip=True, batch_size=1 とする
    """
    tr_opt = argparse.Namespace(**{k: v for k, v in vars(opt).items() if k != 'visualizer'})
    tr_opt.phase = 'train'
    tr_opt.isTrain = False
    tr_opt.serial_batches = True
    tr_opt.no_flip = True
    tr_opt.batch_size = 1
    return tr_opt


def run_validation(model, opt, epoch):
    val_opt = build_val_opt(opt)
    val_dataset = create_dataset(val_opt)

    # set eval
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).eval()

    metrics = MetricsCalculator()
    max_val_samples = min(len(val_dataset), getattr(opt, 'num_val_samples', 500))

    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            if i >= max_val_samples:
                break

            model.set_input(data)
            model.forward()

            # fake_B / real_B は (B*T,C,H,W) に畳まれてるので、ゼロ画素フレームを除外してから metrics に渡す
            fake_B = model.fake_B
            real_B = model.real_B

            if real_B.dim() == 5:
                b, t, c, h, w = real_B.shape
                fake_B_flat = fake_B.view(b * t, c, h, w)
                real_B_flat = real_B.view(b * t, c, h, w)
                valid_idx = get_valid_frame_idx(real_B_flat)
                if valid_idx.numel() > 0 and valid_idx.numel() < real_B_flat.shape[0]:
                    fake_B_flat = fake_B_flat[valid_idx]
                    real_B_flat = real_B_flat[valid_idx]
                fake_use, real_use = fake_B_flat, real_B_flat
            else:
                fake_use, real_use = fake_B, real_B

            metrics.update(fake_use, real_use)

    # back to train
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).train()

    results = metrics.compute()
    print(f'\n[Validation Epoch {epoch}] {metrics}')
    return results


def run_fid(model, opt, epoch):
    val_opt = build_val_opt(opt)
    val_dataset = create_dataset(val_opt)

    # temp dirs
    tmp_gen_dir = tempfile.mkdtemp()
    tmp_gt_dir = tempfile.mkdtemp()

    import torchvision.utils as vutils
    import os

    # eval
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).eval()

    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            model.forward()

            fake_B = model.fake_B.detach().cpu()  # (T,C,H,W) 相当（batch=1前提ならT枚）
            real_B = model.real_B.detach().cpu()

            # ゼロ画素フレームを除外
            if real_B.dim() == 5:
                b, t, c, h, w = real_B.shape
                fake_flat = fake_B.view(b * t, c, h, w)
                real_flat = real_B.view(b * t, c, h, w)
            else:
                fake_flat = fake_B
                real_flat = real_B

            valid_idx = get_valid_frame_idx(real_flat)
            if valid_idx.numel() > 0 and valid_idx.numel() < real_flat.shape[0]:
                fake_flat = fake_flat[valid_idx]
                real_flat = real_flat[valid_idx]

            # 保存（フレーム単位）
            for b in range(fake_flat.size(0)):
                vutils.save_image(fake_flat[b], os.path.join(tmp_gen_dir, f"{i:04d}_{b:02d}.png"), normalize=True)
                vutils.save_image(real_flat[b], os.path.join(tmp_gt_dir, f"{i:04d}_{b:02d}.png"), normalize=True)

    try:
        fid = calculate_fid_given_paths(
            [tmp_gt_dir, tmp_gen_dir],
            batch_size=32,
            cuda=torch.cuda.is_available(),
            dims=2048
        )
    except Exception as e:
        print(f"[FID Calculation Error] {e}")
        fid = float('inf')

    print(f"[Validation Epoch {epoch}] FID: {fid:.4f}")

    shutil.rmtree(tmp_gen_dir)
    shutil.rmtree(tmp_gt_dir)

    # back to train
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).train()

    return fid


def save_ot_snapshots(model, opt, epoch, num_samples=3, epoch_interval=10):
    """10epochごとに、train/val それぞれから数シーケンスを取り出して
    sequence OT の輸送計画を保存する。

    - train 側: build_train_snapshot_opt(opt) でデータセットを構築
    - val 側  : build_val_opt(opt) でデータセットを構築
    - 各 split について先頭 num_samples サンプルを処理
    - P マトリクスと U ベクトルは、モデル側の _save_ot_details で PNG/npz として保存される
    """
    # 引数よりも、オプションで指定された値を優先
    epoch_interval = getattr(opt, 'seq_ot_snapshot_epoch_interval', epoch_interval)
    num_samples = getattr(opt, 'seq_ot_snapshot_num_samples', num_samples)

    if epoch % epoch_interval != 0:
        return

    if not getattr(opt, 'save_ot_details', False):
        return

    # OT 詳細の出力ディレクトリが無ければ何もしない
    if not hasattr(model, 'ot_details_dir'):
        return

    print(f"[OT Snapshot] epoch {epoch}: saving OT details for train/val (num_samples={num_samples})")

    # 一時的に eval モードに切り替え
    nets = []
    for name in model.model_names:
        if isinstance(name, str) and hasattr(model, 'net' + name):
            net = getattr(model, 'net' + name)
            nets.append((name, net.training))
            net.eval()

    try:
        for split in ['train', 'val']:
            if split == 'train':
                snap_opt = build_train_snapshot_opt(opt)
            else:
                snap_opt = build_val_opt(opt)

            dataset = create_dataset(snap_opt)

            # split / epoch ごとにサブディレクトリを切る
            base_dir = os.path.join(model.ot_details_dir, f"{split}_epoch{epoch:04d}")
            os.makedirs(base_dir, exist_ok=True)
            orig_dir = model.ot_details_dir
            model.ot_details_dir = base_dir

            with torch.no_grad():
                for i, data in enumerate(dataset):
                    if i >= num_samples:
                        break

                    # train 側でも sequence OT の可視化だけなので input2 は不要
                    model.set_input(data)
                    model.forward()

                    fake_seq = model.fake_B
                    real_seq = model.real_B

                    if fake_seq.dim() == 5 and real_seq.dim() == 5:
                        b, t, c, h, w = fake_seq.shape
                        # batch_size=1 を想定して先頭のみ
                        for bi in range(min(b, 1)):
                            ot_val, details = sequence_ot_loss_torch(
                                fake_seq[bi], real_seq[bi],
                                    reg=opt.lmda,
                                    iters=getattr(opt, 'seq_ot_iters', 50),
                                    monotone=getattr(opt, 'seq_ot_monotone', True),
                                    monotone_penalty=getattr(opt, 'seq_ot_monotone_penalty', 50.0),
                                    normalize=(None if getattr(opt, 'seq_ot_normalize', 'mean') == 'none' else getattr(opt, 'seq_ot_normalize', 'mean')),
                                return_details=True,
                            )
                            if details is not None:
                                # 対応する realA / realB / fakeB のシーケンスも一緒に保存
                                real_A_seq = None
                                try:
                                    real_A = getattr(model, 'real_A', None)
                                    if real_A is not None:
                                        if real_A.dim() == 5:
                                            real_A_seq = real_A[bi]
                                        else:
                                            real_A_seq = real_A
                                except Exception:
                                    real_A_seq = None

                                real_B_seq = real_seq[bi]
                                fake_B_seq = fake_seq[bi]

                                model._save_ot_details(details, real_A_seq=real_A_seq, real_B_seq=real_B_seq, fake_B_seq=fake_B_seq)
                                if hasattr(model, 'ot_details_saved'):
                                    model.ot_details_saved += 1
                    else:
                        ot_val, details = sequence_ot_loss_torch(
                            fake_seq, real_seq,
                            reg=opt.lmda,
                            iters=getattr(opt, 'seq_ot_iters', 50),
                            monotone=getattr(opt, 'seq_ot_monotone', True),
                            monotone_penalty=getattr(opt, 'seq_ot_monotone_penalty', 50.0),
                            normalize=(None if getattr(opt, 'seq_ot_normalize', 'mean') == 'none' else getattr(opt, 'seq_ot_normalize', 'mean')),
                            return_details=True,
                        )
                        if details is not None:
                            # 4D テンソルの場合も、そのままシーケンスとして保存
                            real_A_seq = None
                            try:
                                real_A = getattr(model, 'real_A', None)
                                if real_A is not None:
                                    real_A_seq = real_A
                            except Exception:
                                real_A_seq = None

                            real_B_seq = real_seq
                            fake_B_seq = fake_seq

                            model._save_ot_details(details, real_A_seq=real_A_seq, real_B_seq=real_B_seq, fake_B_seq=fake_B_seq)
                            if hasattr(model, 'ot_details_saved'):
                                model.ot_details_saved += 1

            # 元のディレクトリに戻す
            model.ot_details_dir = orig_dir
    finally:
        # 元の train/eval 状態に戻す
        for name, was_training in nets:
            net = getattr(model, 'net' + name)
            net.train(was_training)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)

    model = create_model(opt)
    print('The number of training sequences = %d' % dataset_size)

    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0
    optimize_time = 0.1

    best_fid = float('inf')
    best_epoch = 0

    # 学習開始前の初期状態 (epoch=0) についても、固定シーケンスの OT 詳細を保存しておく
    # （0, 10, 20, ... のように揃えて可視化できるようにする）
    if getattr(opt, 'save_ot_details', False) and not getattr(opt, 'continue_train', False):
        save_ot_snapshots(model, opt, epoch=0, num_samples=3, epoch_interval=10)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        dataset.set_epoch(epoch)
        dataset2.set_epoch(epoch)

        for i, (data, data2) in enumerate(zip(dataset, dataset2)):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # data["A"] は (B,T,1,H,W) なので batch_size=B
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()

            if epoch == opt.epoch_count and i == 0:
                # ver1 と同様に、data/data2 の 2 本を使って
                # netF 初期化や SB 関連の統計を安定化させる
                model.data_dependent_initialize(data, data2)
                model.setup(opt)
                model.parallelize()
                
            model.set_input(data, data2)
            model.optimize_parameters()

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / max(batch_size, 1) * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                with torch.no_grad():
                    model.compute_visuals()
                    visualizer.display_current_results(
                        model.get_current_visuals(),
                        epoch,
                        save_result,
                        global_step=total_iters,
                        split='train',
                    )

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, optimize_time, t_data,
                    global_step=total_iters,
                    split='train',
                )
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # validation frequency
        val_freq = getattr(opt, 'val_epoch_freq', opt.save_epoch_freq)
        if epoch % val_freq == 0:
            val_metrics = run_validation(model, opt, epoch)
            visualizer.print_current_losses(
                epoch,
                0,
                {
                    'val_SSIM': val_metrics['SSIM'],
                    'val_L2': val_metrics['L2'],
                    'val_PSNR': val_metrics['PSNR'],
                },
                0,
                0,
                global_step=total_iters,
                split='val',
            )
            if opt.display_id is None or opt.display_id > 0:
                visualizer.plot_current_losses(epoch, 1.0, {'val_SSIM': val_metrics['SSIM'], 'val_L2': val_metrics['L2']})

            fid = run_fid(model, opt, epoch)

            # log a single validation sample visuals to W&B (optional)
            if getattr(opt, 'use_wandb', False):
                try:
                    val_opt = build_val_opt(opt)
                    val_dataset = create_dataset(val_opt)
                    with torch.no_grad():
                        for j, vdata in enumerate(val_dataset):
                            model.set_input(vdata)
                            model.forward()
                            model.compute_visuals()
                            visualizer.wandb_log_visuals(
                                model.get_current_visuals(),
                                epoch,
                                global_step=total_iters,
                                split='val',
                            )
                            break
                except Exception as e:
                    print(f"[wandb][val_visuals] failed: {e}")

            # log FID into the same loss_log.txt format for plotting
            visualizer.print_current_losses(
                epoch,
                0,
                {'val_FID': fid},
                0,
                0,
                global_step=total_iters,
                split='val',
            )

            # 10epochごとに、train/val それぞれから同じシーケンス数本について
            # sequence OT の輸送計画を保存（P マトリクス等を PNG/npz として出力）
            save_ot_snapshots(model, opt, epoch, num_samples=3, epoch_interval=10)

            # If OT snapshot images exist, log a few to W&B
            if getattr(opt, 'use_wandb', False) and hasattr(model, 'ot_details_dir'):
                try:
                    ot_epoch_dirs = [
                        os.path.join(model.ot_details_dir, f"train_epoch{epoch:04d}"),
                        os.path.join(model.ot_details_dir, f"val_epoch{epoch:04d}"),
                    ]
                    ot_images = []
                    for d in ot_epoch_dirs:
                        if not os.path.isdir(d):
                            continue
                        # Prefer the most informative images
                        for suffix in ['_P.png', '_M.png', '_U.png', '_real_A.png', '_fake_B.png', '_real_B.png']:
                            ot_images.extend(sorted(
                                [os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(suffix)]
                            ))
                    # cap to avoid huge uploads
                    ot_images = ot_images[:24]
                    if ot_images:
                        visualizer.wandb_log_files_as_images(
                            ot_images,
                            global_step=total_iters,
                            split='val',
                            prefix=f"ot_epoch{epoch:04d}",
                            epoch=epoch,
                        )
                except Exception as e:
                    print(f"[wandb][ot] failed: {e}")

            # best = 最小FID
            if fid < best_fid:
                best_fid = fid
                best_epoch = epoch
                print(f'[Best Model] New best FID: {best_fid:.4f} at epoch {epoch}')
                model.save_networks('best')

                if getattr(opt, 'use_wandb', False):
                    visualizer.print_current_losses(
                        epoch,
                        0,
                        {'val_best_FID': best_fid, 'val_best_epoch': best_epoch},
                        0,
                        0,
                        global_step=total_iters,
                        split='val',
                    )

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        # log current LR for plotting / W&B
        try:
            lr = model.optimizers[0].param_groups[0]['lr']
            visualizer.print_current_losses(
                epoch,
                0,
                {'lr': lr},
                0,
                0,
                global_step=total_iters,
                split='train',
            )
        except Exception:
            pass

    print(f'\n{"="*50}')
    print('Training Complete!')
    print(f'Best Model: Epoch {best_epoch} with FID {best_fid:.4f}')
    print(f'{"="*50}')