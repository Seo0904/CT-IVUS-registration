#!/usr/bin/env python3
"""
Post-hoc EMA Builder:
Runs training from existing checkpoint for N steps to accumulate EMA weights,
then saves them for use in test/validate.

Usage:
    python3 build_ema.py --steps 500 --ema_decay 0.999
"""
import sys, os, time, argparse
sys.path.insert(0, '/workspace/WP-UNSB')

import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.ema import EMA


def main():
    # Extra args for EMA building
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--ema_steps', type=int, default=500, help='number of training steps for EMA accumulation')
    extra_parser.add_argument('--ema_warmup', type=int, default=20, help='steps before starting EMA (to escape dark state)')
    extra_args, remaining = extra_parser.parse_known_args()

    # Replace sys.argv for TrainOptions
    sys.argv = ['build_ema.py'] + remaining

    opt = TrainOptions().parse()
    opt.use_ema = True  # Force EMA on
    dataset = create_dataset(opt)
    model = create_model(opt)

    data_iter = iter(dataset)
    data = next(data_iter)

    model.data_dependent_initialize(data, None)
    model.setup(opt)
    model.parallelize()

    ema_decay = getattr(opt, 'ema_decay', 0.999)
    ema = EMA(model.netG, decay=ema_decay)

    total_steps = extra_args.ema_steps
    warmup_steps = extra_args.ema_warmup

    print(f"\n{'='*60}")
    print(f"Building EMA weights: {total_steps} steps, decay={ema_decay}")
    print(f"Warmup (pre-EMA training): {warmup_steps} steps")
    print(f"{'='*60}\n")

    # Warmup: train for a few steps to escape dark state BEFORE starting EMA
    print("[Phase 1] Warmup training...")
    for step in range(warmup_steps):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            data = next(data_iter)

        model.set_input(data)
        model.optimize_parameters()

        if step % 10 == 0:
            fake = model.fake_B.detach().cpu()
            mean_val = ((fake + 1) / 2 * 255).mean().item()
            print(f"  Warmup step {step:3d}: fake_B mean = {mean_val:.1f}")

    # Now register EMA with the warmed-up weights
    ema.register()
    print(f"\n[Phase 2] Starting EMA accumulation for {total_steps} steps...")

    for step in range(total_steps):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            data = next(data_iter)

        model.set_input(data)
        model.optimize_parameters()
        ema.update()

        if step % 50 == 0:
            fake = model.fake_B.detach().cpu()
            mean_normal = ((fake + 1) / 2 * 255).mean().item()

            # Check EMA output
            ema.apply_shadow()
            model.set_input(data)
            with torch.no_grad():
                model.forward()
            fake_ema = model.fake_B.detach().cpu()
            mean_ema = ((fake_ema + 1) / 2 * 255).mean().item()
            ema.restore()

            print(f"  Step {step:4d}: Normal={mean_normal:.1f}, EMA={mean_ema:.1f}")

    # Save EMA weights
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    ema_path = os.path.join(save_dir, 'latest_net_G_ema.pth')
    ema.save(ema_path)
    print(f"\n{'='*60}")
    print(f"EMA weights saved to: {ema_path}")

    # Also save as 'best_net_G_ema.pth'
    ema_best_path = os.path.join(save_dir, 'best_net_G_ema.pth')
    ema.save(ema_best_path)
    print(f"EMA weights also saved to: {ema_best_path}")

    # Final test
    ema.apply_shadow()
    model.set_input(data)
    with torch.no_grad():
        model.forward()
    fake_final = model.fake_B.detach().cpu()
    mean_final = ((fake_final + 1) / 2 * 255).mean().item()
    ema.restore()
    print(f"\nFinal EMA output: fake_B mean = {mean_final:.1f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
