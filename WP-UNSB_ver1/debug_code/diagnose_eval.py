"""
eval/train mode による生成画像の違いを診断するスクリプト
"""
import sys
sys.path.insert(0, '/workspace/WP-UNSB')
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util as util_mod

def main():
    # テストと同じオプションでモデルを作成
    cmd = (
        "--dataroot /workspace/data/org_data/moving_mnist "
        "--dataroot_B /workspace/data/preprocessed/bspline_transformed "
        "--data_file_A mnist_test_seq.npy "
        "--data_file_B transformed_aligned.npy "
        "--name moving_mnist_paired_sb "
        "--model wpsb "
        "--mode sb "
        "--dataset_mode moving_mnist_paired "
        "--input_nc 1 --output_nc 1 --ngf 64 --ndf 64 "
        "--num_timesteps 5 "
        "--load_size 64 --crop_size 64 "
        "--gpu_ids 0 "
        "--checkpoints_dir /workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347 "
        "--phase test "
        "--epoch latest "
        "--num_test 5 "
        "--eval "
        "--no_flip --serial_batches"
    )
    
    opt = TestOptions(cmd).parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    print("="*60)
    print("DIAGNOSTIC: Train vs Eval mode comparison")
    print("="*60)
    
    # データセット作成
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    
    # モデル作成
    model = create_model(opt)
    
    # 最初のデータで初期化
    for i, (data, data2) in enumerate(zip(dataset, dataset2)):
        if i == 0:
            first_data = data
            first_data2 = data2
            break
    
    model.data_dependent_initialize(first_data, first_data2)
    model.setup(opt)
    model.parallelize()
    
    print("\n--- Model loaded successfully ---")
    print(f"model_names: {model.model_names}")
    print(f"visual_names: {model.visual_names}")
    print(f"isTrain: {model.isTrain}")
    print(f"opt.no_dropout: {opt.no_dropout}")
    
    # データ確認
    print("\n--- Data inspection ---")
    data = first_data
    A = data['A']
    B = data['B']
    print(f"data['A'] shape: {A.shape}, dtype: {A.dtype}")
    print(f"data['A'] min/max/mean: {A.min().item():.4f} / {A.max().item():.4f} / {A.mean().item():.4f}")
    print(f"data['B'] shape: {B.shape}, dtype: {B.dtype}")
    print(f"data['B'] min/max/mean: {B.min().item():.4f} / {B.max().item():.4f} / {B.mean().item():.4f}")
    
    # 固定入力＆ノイズで比較
    torch.manual_seed(42)
    
    # ===== TRAIN mode =====
    print("\n--- TRAIN MODE ---")
    model.netG.train()
    model.set_input(data, first_data2)
    
    print(f"real_A shape: {model.real_A.shape}")
    print(f"real_A min/max/mean: {model.real_A.min().item():.4f} / {model.real_A.max().item():.4f} / {model.real_A.mean().item():.4f}")
    print(f"real_B shape: {model.real_B.shape}")
    print(f"real_B min/max/mean: {model.real_B.min().item():.4f} / {model.real_B.max().item():.4f} / {model.real_B.mean().item():.4f}")
    
    # 決定論的forward
    torch.manual_seed(42)
    with torch.no_grad():
        model.forward()
    
    fake_train = model.fake_B.detach().clone()
    print(f"fake_B (train) shape: {fake_train.shape}")
    print(f"fake_B (train) min/max/mean: {fake_train.min().item():.4f} / {fake_train.max().item():.4f} / {fake_train.mean().item():.4f}")
    
    # ===== EVAL mode =====
    print("\n--- EVAL MODE ---")
    model.netG.eval()
    model.set_input(data, first_data2)
    
    # 同じシードで同じノイズ
    torch.manual_seed(42)
    with torch.no_grad():
        model.forward()
    
    fake_eval = model.fake_B.detach().clone()
    print(f"fake_B (eval) shape: {fake_eval.shape}")
    print(f"fake_B (eval) min/max/mean: {fake_eval.min().item():.4f} / {fake_eval.max().item():.4f} / {fake_eval.mean().item():.4f}")
    
    # 差分
    diff = (fake_train - fake_eval).abs()
    print(f"\n--- Comparison ---")
    print(f"diff min/max/mean: {diff.min().item():.6f} / {diff.max().item():.6f} / {diff.mean().item():.6f}")
    
    # 黒いかどうかのチェック（値が-1に近い=黒）
    near_minus1_train = (fake_train < -0.9).float().mean().item()
    near_minus1_eval = (fake_eval < -0.9).float().mean().item()
    print(f"\nPixels near -1 (black): train={near_minus1_train:.4f}, eval={near_minus1_eval:.4f}")
    
    # tensor2im の結果確認
    train_im = util_mod.tensor2im(fake_train[0])
    eval_im = util_mod.tensor2im(fake_eval[0])
    print(f"\ntensor2im result (train): shape={train_im.shape}, min={train_im.min()}, max={train_im.max()}, mean={train_im.mean():.1f}")
    print(f"tensor2im result (eval):  shape={eval_im.shape}, min={eval_im.min()}, max={eval_im.max()}, mean={eval_im.mean():.1f}")
    
    # 各レイヤーのdropout確認
    print("\n--- Layer analysis ---")
    for name, mod in model.netG.named_modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout2d)):
            print(f"  DROPOUT FOUND: {name} p={mod.p} training={mod.training}")
        if isinstance(mod, torch.nn.BatchNorm2d):
            print(f"  BATCHNORM FOUND: {name} track_running_stats={mod.track_running_stats} training={mod.training}")
        if isinstance(mod, torch.nn.InstanceNorm2d):
            if mod.affine or mod.track_running_stats:
                print(f"  INSTANCENORM (non-default): {name} affine={mod.affine} track_running_stats={mod.track_running_stats}")
    
    print("\n--- Diagnosis complete ---")

if __name__ == '__main__':
    main()
