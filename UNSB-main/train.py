import time
import torch
import argparse
import shutil
import tempfile
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.metrics import MetricsCalculator
from vgg_sb.evaluations.fid_score import calculate_fid_given_paths


def build_val_opt(opt):
    """Create validation options (phase='val', no aug, batch_size=1)."""
    val_opt = argparse.Namespace(**{k: v for k, v in vars(opt).items() if k != 'visualizer'})
    val_opt.phase = 'val'
    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.batch_size = 1
    return val_opt


def run_validation(model, opt, epoch):
    """Run validation and compute SSIM/L2/PSNR (no FID here)."""
    val_opt = build_val_opt(opt)
    val_dataset = create_dataset(val_opt)

    # Set networks to eval mode
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).eval()

    metrics = MetricsCalculator()
    max_val_samples = min(len(val_dataset), getattr(opt, 'num_val_samples', 500))

    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            if i >= max_val_samples:
                break

            # Validation uses paired data; ver1 形式と同様に単一入力で実行
            model.set_input(data)
            model.forward()

            # fake_B / real_B は (B*T,C,H,W) 相当を想定
            metrics.update(model.fake_B, model.real_B)

    # Set networks back to train mode
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, 'net' + name).train()

    results = metrics.compute()
    print(f'\n[Validation Epoch {epoch}] {metrics}')
    return results


def run_fid(model, opt, epoch):
    """Compute FID using temporary PNGs and fid_score.calculate_fid_given_paths.

    This follows the same procedure as WP-UNSB_ver1/train.py for fair comparison.
    """
    val_opt = build_val_opt(opt)
    val_dataset = create_dataset(val_opt)

    # Temporary directories for generated and GT images
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

            fake_B = model.fake_B.detach().cpu()
            real_B = model.real_B.detach().cpu()

            # 保存（フレーム単位）: fake/real をそれぞれ PNG で保存
            for b in range(fake_B.size(0)):
                vutils.save_image(fake_B[b], os.path.join(tmp_gen_dir, f"{i:04d}_{b:02d}.png"), normalize=True)
                vutils.save_image(real_B[b], os.path.join(tmp_gt_dir, f"{i:04d}_{b:02d}.png"), normalize=True)

    try:
        fid = calculate_fid_given_paths(
            [tmp_gt_dir, tmp_gen_dir],
            batch_size=32,
            cuda=torch.cuda.is_available(),
            dims=2048,
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
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    # Best model tracking (based on lowest FID on validation data)
    best_fid = float('inf')
    best_epoch = 0

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, (data,data2) in enumerate(zip(dataset,dataset2)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data,data2)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data,data2)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Run validation at the end of each epoch (or at specified frequency)
        val_freq = opt.val_epoch_freq if hasattr(opt, 'val_epoch_freq') else opt.save_epoch_freq
        if epoch % val_freq == 0:
            val_metrics = run_validation(model, opt, epoch)
            # Log validation metrics (SSIM/L2/PSNR)
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
            )
            if opt.display_id is None or opt.display_id > 0:
                visualizer.plot_current_losses(
                    epoch,
                    1.0,
                    {
                        'val_SSIM': val_metrics['SSIM'],
                        'val_L2': val_metrics['L2'],
                    },
                )

            # Compute FID using PNG-based pipeline (same as WP-UNSB)
            fid = run_fid(model, opt, epoch)

            # Save best model based on FID (lower is better)
            if fid < best_fid:
                best_fid = fid
                best_epoch = epoch
                print(f'[Best Model] New best FID: {best_fid:.4f} at epoch {epoch}')
                model.save_networks('best')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    # Print final best model info
    print(f'\n{"="*50}')
    print('Training Complete!')
    print(f'Best Model: Epoch {best_epoch} with FID {best_fid:.4f}')
    print(f'{"="*50}')
