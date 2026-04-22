"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import json
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from util.metrics import MetricsCalculator, ssim, l2_loss, l1_loss, psnr


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # Initialize metrics calculator
    metrics = MetricsCalculator()
    all_metrics = []  # Store per-sample metrics

    for i, (data,data2) in enumerate(zip(dataset,dataset2)):
        if i == 0:
            model.data_dependent_initialize(data,data2)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data,data2)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        # Calculate metrics if we have paired data (fake_B vs real_B)
        if hasattr(model, 'fake_B') and hasattr(model, 'real_B'):
            fake_B = model.fake_B
            real_B = model.real_B
            
            sample_metrics = {
                'index': i,
                'path': img_path[0] if isinstance(img_path, list) else str(img_path),
                'SSIM': ssim(fake_B, real_B),
                'L2': l2_loss(fake_B, real_B),
                'L1': l1_loss(fake_B, real_B),
                'PSNR': psnr(fake_B, real_B)
            }
            all_metrics.append(sample_metrics)
            metrics.update(fake_B, real_B)
        
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    
    webpage.save()  # save the HTML
    
    # Save metrics results
    if all_metrics:
        import numpy as np
        
        final_metrics = metrics.compute()
        
        # Print summary
        print('\n' + '='*50)
        print('TEST RESULTS')
        print('='*50)
        print(f'Samples evaluated: {len(all_metrics)}')
        print(metrics)
        
        # Calculate statistics
        ssim_values = [m['SSIM'] for m in all_metrics]
        l2_values = [m['L2'] for m in all_metrics]
        l1_values = [m['L1'] for m in all_metrics]
        psnr_values = [m['PSNR'] for m in all_metrics]
        
        results_summary = {
            'epoch': opt.epoch,
            'num_samples': len(all_metrics),
            'metrics': {
                'SSIM': {
                    'mean': float(np.mean(ssim_values)),
                    'std': float(np.std(ssim_values)),
                    'min': float(np.min(ssim_values)),
                    'max': float(np.max(ssim_values))
                },
                'L2': {
                    'mean': float(np.mean(l2_values)),
                    'std': float(np.std(l2_values)),
                    'min': float(np.min(l2_values)),
                    'max': float(np.max(l2_values))
                },
                'L1': {
                    'mean': float(np.mean(l1_values)),
                    'std': float(np.std(l1_values)),
                    'min': float(np.min(l1_values)),
                    'max': float(np.max(l1_values))
                },
                'PSNR': {
                    'mean': float(np.mean(psnr_values)),
                    'std': float(np.std(psnr_values)),
                    'min': float(np.min(psnr_values)),
                    'max': float(np.max(psnr_values))
                }
            },
            'per_sample': all_metrics
        }
        
        # Save to JSON
        metrics_file = os.path.join(web_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f'\nMetrics saved to: {metrics_file}')
        
        # Save to text file
        txt_file = os.path.join(web_dir, 'metrics.txt')
        with open(txt_file, 'w') as f:
            f.write(f'Test Results - Epoch {opt.epoch}\n')
            f.write('='*50 + '\n\n')
            f.write(f'Samples evaluated: {len(all_metrics)}\n\n')
            for metric_name in ['SSIM', 'L2', 'L1', 'PSNR']:
                stats = results_summary['metrics'][metric_name]
                f.write(f'{metric_name}:\n')
                f.write(f'  Mean: {stats["mean"]:.6f} ± {stats["std"]:.6f}\n')
                f.write(f'  Range: [{stats["min"]:.6f}, {stats["max"]:.6f}]\n\n')
            
            # Write per-sample metrics
            f.write('Per-sample metrics:\n')
            f.write('='*50 + '\n')
            for m in all_metrics:
                f.write(f'Image {m["index"]} - Path: {m["path"]}\n')
                f.write(f'  SSIM: {m["SSIM"]:.6f}\n')
                f.write(f'  L2: {m["L2"]:.6f}\n')
                f.write(f'  L1: {m["L1"]:.6f}\n')
                f.write(f'  PSNR: {m["PSNR"]:.6f}\n')
                f.write('\n')
        print(f'Metrics saved to: {txt_file}')
        print('='*50)
