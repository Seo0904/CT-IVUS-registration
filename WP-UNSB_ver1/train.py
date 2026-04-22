# =========================
# 4) train.py（全部反映版）
#    - dataset2/zipを削除（set_inputがinput2使ってないので無意味）
#    - validation と FID は必ず val_opt (phase='val') を使う
#    - best は FID最小
# =========================
import time
import torch
import argparse
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from util.metrics import MetricsCalculator
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

            # fake_B / real_B は (B*T,C,H,W) に畳まれてるので、そのままmetricsに突っ込める
            metrics.update(model.fake_B, model.real_B)

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

            # 保存（フレーム単位）
            for b in range(fake_B.size(0)):
                vutils.save_image(fake_B[b], os.path.join(tmp_gen_dir, f"{i:04d}_{b:02d}.png"), normalize=True)
                vutils.save_image(real_B[b], os.path.join(tmp_gt_dir, f"{i:04d}_{b:02d}.png"), normalize=True)

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


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    model = create_model(opt)
    print('The number of training sequences = %d' % dataset_size)

    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0
    optimize_time = 0.1

    best_fid = float('inf')
    best_epoch = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
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
                model.data_dependent_initialize(data, None)
                model.setup(opt)
                model.parallelize()
                model.init_ema()  # Initialize EMA after model is fully set up

            model.set_input(data)
            model.optimize_parameters()

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / max(batch_size, 1) * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                with torch.no_grad():
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
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
                epoch, 0,
                {'val_SSIM': val_metrics['SSIM'], 'val_L2': val_metrics['L2'], 'val_PSNR': val_metrics['PSNR']},
                0, 0
            )
            if opt.display_id is None or opt.display_id > 0:
                visualizer.plot_current_losses(epoch, 1.0, {'val_SSIM': val_metrics['SSIM'], 'val_L2': val_metrics['L2']})

            fid = run_fid(model, opt, epoch)

            # best = 最小FID
            if fid < best_fid:
                best_fid = fid
                best_epoch = epoch
                print(f'[Best Model] New best FID: {best_fid:.4f} at epoch {epoch}')
                model.save_networks('best')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    print(f'\n{"="*50}')
    print('Training Complete!')
    print(f'Best Model: Epoch {best_epoch} with FID {best_fid:.4f}')
    print(f'{"="*50}')