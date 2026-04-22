"""
train.pyとtest.pyのモデル推論フローの違いを正確に再現して比較
"""
import sys
sys.path.insert(0, '/workspace/WP-UNSB')
import torch
import numpy as np
import argparse
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

CKPT_DIR = '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347'

def get_train_model_and_data():
    """train.pyと同じフローでモデルを作って学習重みを読み込む"""
    cmd = (
        '--dataroot /workspace/data/org_data/moving_mnist '
        '--dataroot_B /workspace/data/preprocessed/bspline_transformed '
        '--data_file_A mnist_test_seq.npy '
        '--data_file_B transformed_aligned.npy '
        '--name moving_mnist_paired_sb '
        '--model wpsb --mode sb '
        '--dataset_mode moving_mnist_paired '
        '--input_nc 1 --output_nc 1 --ngf 64 --ndf 64 '
        '--num_timesteps 5 '
        '--load_size 64 --crop_size 64 --gpu_ids 0 '
        f'--checkpoints_dir {CKPT_DIR} '
        '--batch_size 8 '
        '--preprocess none --no_flip '
        '--n_epochs 200 --n_epochs_decay 200 '
        '--lr 0.00001 --save_epoch_freq 10 '
        '--lambda_GAN 1.0 --lambda_SB 1.0 --lambda_NCE 1.0 '
        '--continue_train '
        '--epoch latest'
    )
    opt = TrainOptions(cmd).parse()
    dataset = create_dataset(opt)
    model = create_model(opt)
    return model, dataset, opt


def get_test_model_and_data():
    """test.pyと同じフローでモデルを作る"""
    cmd = (
        '--dataroot /workspace/data/org_data/moving_mnist '
        '--dataroot_B /workspace/data/preprocessed/bspline_transformed '
        '--data_file_A mnist_test_seq.npy '
        '--data_file_B transformed_aligned.npy '
        '--name moving_mnist_paired_sb '
        '--model wpsb --mode sb '
        '--dataset_mode moving_mnist_paired '
        '--input_nc 1 --output_nc 1 --ngf 64 --ndf 64 '
        '--num_timesteps 5 '
        '--load_size 64 --crop_size 64 --gpu_ids 0 '
        f'--checkpoints_dir {CKPT_DIR} '
        '--phase test --epoch latest '
        '--num_test 5 --eval --no_flip --serial_batches'
    )
    opt = TestOptions(cmd).parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    model = create_model(opt)
    return model, dataset, dataset2, opt


def main():
    print("="*60)
    print("Step 1: Create TRAIN model (same as train.py)")
    print("="*60)
    
    train_model, train_dataset, train_opt = get_train_model_and_data()
    
    # train.pyのフローを再現: 最初のデータでDDI
    first_data = None
    for data in train_dataset:
        first_data = data
        break
    
    train_model.data_dependent_initialize(first_data, None)
    train_model.setup(train_opt)  # これでcheckpointが読み込まれる (continue_train=True)
    train_model.parallelize()
    
    print("\n--- Train model state ---")
    net = train_model.netG
    if hasattr(net, 'module'): net = net.module
    print(f"netG.training: {net.training}")
    print(f"isTrain: {train_model.isTrain}")
    print(f"model_names: {train_model.model_names}")
    
    # train forwardを再現（optimize_parametersの中のforward）
    print("\n--- Train forward (like optimize_parameters) ---")
    train_model.netG.train()    
    for i, data in enumerate(train_dataset):
        if i >= 3: break
        train_model.set_input(data)
        
        # optimize_parametersの最初のforwardと同じ
        with torch.no_grad():
            train_model.forward()
        
        fake = train_model.fake_B.detach()
        real_A = train_model.real_A.detach()
        
        fm = ((fake.cpu().float() + 1) / 2 * 255).mean().item()
        rm = ((real_A.cpu().float() + 1) / 2 * 255).mean().item()
        ratio = fm / max(rm, 0.01)
        
        # batch_size=8なのでreal_Aは (8*20, 1, 64, 64) = (160, 1, 64, 64)
        print(f"  Train sample {i}: real_A shape={tuple(real_A.shape)}, fake shape={tuple(fake.shape)}")
        print(f"  Train sample {i}: fake_mean={fm:.1f}, realA_mean={rm:.1f}, ratio={ratio:.3f}")
    
    # ============================================================
    print("\n" + "="*60)
    print("Step 2: Create TEST model (same as test.py)")
    print("="*60)
    
    test_model, test_dataset, test_dataset2, test_opt = get_test_model_and_data()
    
    for i, (data, data2) in enumerate(zip(test_dataset, test_dataset2)):
        if i == 0:
            test_model.data_dependent_initialize(data, data2)
            test_model.setup(test_opt)
            test_model.parallelize()
            if test_opt.eval:
                test_model.eval()
        if i >= 3: break
        test_model.set_input(data, data2)
        test_model.test()
        
        fake = test_model.fake_B.detach()
        real_A = test_model.real_A.detach()
        
        fm = ((fake.cpu().float() + 1) / 2 * 255).mean().item()
        rm = ((real_A.cpu().float() + 1) / 2 * 255).mean().item()
        ratio = fm / max(rm, 0.01)
        
        print(f"  Test sample {i}: real_A shape={tuple(real_A.shape)}, fake shape={tuple(fake.shape)}")
        print(f"  Test sample {i}: fake_mean={fm:.1f}, realA_mean={rm:.1f}, ratio={ratio:.3f}")
    
    # ============================================================
    print("\n" + "="*60)
    print("Step 3: Use TRAIN model with TRAIN data but batch_size=1")
    print("="*60)
    
    # batch_size=1のtrainデータでtrain modelを使う
    for i, data in enumerate(train_dataset):
        if i >= 3: break
        # batch_size=8のうち最初の1シーケンスだけ使う
        single_data = {
            'A': data['A'][:1],
            'B': data['B'][:1],
            'A_paths': data['A_paths'],
            'B_paths': data['B_paths'],
        }
        train_model.set_input(single_data)
        with torch.no_grad():
            train_model.forward()
        
        fake = train_model.fake_B.detach()
        real_A = train_model.real_A.detach()
        
        fm = ((fake.cpu().float() + 1) / 2 * 255).mean().item()
        rm = ((real_A.cpu().float() + 1) / 2 * 255).mean().item()
        ratio = fm / max(rm, 0.01)
        
        print(f"  TrainModel bs=1 sample {i}: fake_mean={fm:.1f}, realA_mean={rm:.1f}, ratio={ratio:.3f}")
    
    # ============================================================
    print("\n" + "="*60)
    print("Step 4: Use TEST model with TRAIN data")
    print("="*60)
    
    # テストモデルでtrainsのデータを使う
    train_cmd = (
        '--dataroot /workspace/data/org_data/moving_mnist '
        '--dataroot_B /workspace/data/preprocessed/bspline_transformed '
        '--data_file_A mnist_test_seq.npy '
        '--data_file_B transformed_aligned.npy '
        '--name moving_mnist_paired_sb '
        '--model wpsb --mode sb '
        '--dataset_mode moving_mnist_paired '
        '--input_nc 1 --output_nc 1 --ngf 64 --ndf 64 '
        '--num_timesteps 5 '
        '--load_size 64 --crop_size 64 --gpu_ids 0 '
        f'--checkpoints_dir {CKPT_DIR} '
        '--phase train --epoch latest '
        '--num_test 5 --eval --no_flip --serial_batches'
    )
    train_as_test_opt = TestOptions(train_cmd).parse()
    train_as_test_opt.num_threads = 0
    train_as_test_opt.batch_size = 1
    train_as_test_dataset = create_dataset(train_as_test_opt)
    
    for i, data in enumerate(train_as_test_dataset):
        if i >= 3: break
        test_model.set_input(data)
        test_model.test()
        
        fake = test_model.fake_B.detach()
        real_A = test_model.real_A.detach()
        
        fm = ((fake.cpu().float() + 1) / 2 * 255).mean().item()
        rm = ((real_A.cpu().float() + 1) / 2 * 255).mean().item()
        ratio = fm / max(rm, 0.01)
        
        print(f"  TestModel+TrainData sample {i}: fake_mean={fm:.1f}, realA_mean={rm:.1f}, ratio={ratio:.3f}")

    # ============================================================
    print("\n" + "="*60)
    print("Step 5: Weight comparison")
    print("="*60)
    
    # 重みが実際に同じか確認
    net_train = train_model.netG
    net_test = test_model.netG
    if hasattr(net_train, 'module'): net_train = net_train.module
    if hasattr(net_test, 'module'): net_test = net_test.module
    
    total_diff = 0
    total_params = 0
    for (n1, p1), (n2, p2) in zip(net_train.named_parameters(), net_test.named_parameters()):
        diff = (p1.data - p2.data).abs().sum().item()
        total_diff += diff
        total_params += p1.numel()
    
    print(f"Total weight diff: {total_diff:.6f}")
    print(f"Total params: {total_params}")
    print(f"Avg diff per param: {total_diff/total_params:.10f}")


if __name__ == '__main__':
    main()
