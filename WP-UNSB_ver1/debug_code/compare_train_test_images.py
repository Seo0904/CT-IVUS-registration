"""
train/testの保存画像を直接比較する
"""
from PIL import Image
import numpy as np
import os

train_dir = '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347/moving_mnist_paired_sb/web/images'
test_dir = '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347/test_results/moving_mnist_paired_sb/test_best/images'

# ============================================================
# TRAIN visuals: epoch_XXX_fake_B_seq.png は3フレーム横連結 (64, 192, 3)
# TEST visuals: fake_B/seqX_frameY.png は1フレーム (64, 64, 3) or (64, 192, 3)
# ============================================================

# サンプル調査: test画像のサイズ確認
test_fake_dir = os.path.join(test_dir, 'fake_B')
test_realA_dir = os.path.join(test_dir, 'real_A')
test_realB_dir = os.path.join(test_dir, 'real_B')

# TEST画像のフォーマット確認
sample = Image.open(os.path.join(test_fake_dir, 'seq0_frame0.png'))
print(f"Test fake_B sample: size={sample.size}, mode={sample.mode}")
test_arr = np.array(sample)
print(f"  array shape: {test_arr.shape}")

# TRAIN画像のフォーマット確認  
train_sample = Image.open(os.path.join(train_dir, 'epoch400_fake_B_seq.png'))
print(f"Train fake_B_seq sample: size={train_sample.size}, mode={train_sample.mode}")
train_arr = np.array(train_sample)
print(f"  array shape: {train_arr.shape}")

# 3枚横連結なので各1/3を比較するのが公平
# train: (64, 192, 3) -> 3枚の (64, 64, 3) 
# test: fake_B/seqX_frameY.png は (64, 192, 3)? or (64, 64, 3)?

print("\n" + "="*60)
print("FAIR COMPARISON: Non-zero pixel ratio")
print("="*60)

# TRAIN (last 100 epochs)
print("\n--- TRAIN fake_B_seq (last 100 epochs) ---")
train_nonzero = []
train_means = []
train_ratios = []
for epoch in range(301, 401):
    fake_path = os.path.join(train_dir, f'epoch{epoch:03d}_fake_B_seq.png')
    realA_path = os.path.join(train_dir, f'epoch{epoch:03d}_real_A_seq.png')
    if not os.path.exists(fake_path): continue
    fake = np.array(Image.open(fake_path)).astype(float)
    realA = np.array(Image.open(realA_path)).astype(float)
    # ピクセルが>30の割合（白い数字部分）
    fake_bright = (fake > 30).sum() / fake.size * 100
    realA_bright = (realA > 30).sum() / realA.size * 100
    train_nonzero.append(fake_bright)
    train_means.append(fake.mean())
    if realA_bright > 0:
        train_ratios.append(fake_bright / realA_bright)

train_nonzero = np.array(train_nonzero)
train_means = np.array(train_means)
train_ratios = np.array(train_ratios)
print(f"  Bright pixel % (>30): mean={train_nonzero.mean():.2f}%, std={train_nonzero.std():.2f}%")
print(f"  Mean value: mean={train_means.mean():.1f}, std={train_means.std():.1f}")
print(f"  bright ratio (fake/real): mean={train_ratios.mean():.3f}")

# TEST (first 200 sequences, all frames)
print("\n--- TEST fake_B (200 seqs, all frames) ---")
test_nonzero = []
test_means_list = []
test_bright_ratios = []
for seq in range(200):
    for frame in range(20):
        fname = f'seq{seq}_frame{frame}.png'
        fp = os.path.join(test_fake_dir, fname)
        rp = os.path.join(test_realA_dir, fname)
        if not os.path.exists(fp): continue
        fake = np.array(Image.open(fp)).astype(float)
        realA = np.array(Image.open(rp)).astype(float)
        fake_bright = (fake > 30).sum() / fake.size * 100
        realA_bright = (realA > 30).sum() / realA.size * 100
        test_nonzero.append(fake_bright)
        test_means_list.append(fake.mean())
        if realA_bright > 0:
            test_bright_ratios.append(fake_bright / realA_bright)

test_nonzero = np.array(test_nonzero)
test_means_arr = np.array(test_means_list)
test_bright_ratios = np.array(test_bright_ratios)
print(f"  Bright pixel % (>30): mean={test_nonzero.mean():.2f}%, std={test_nonzero.std():.2f}%")
print(f"  Mean value: mean={test_means_arr.mean():.1f}, std={test_means_arr.std():.1f}")
print(f"  bright ratio (fake/real): mean={test_bright_ratios.mean():.3f}")

print(f"\n=== CONCLUSION ===")
print(f"  TRAIN bright pixels: {train_nonzero.mean():.2f}%")
print(f"  TEST  bright pixels: {test_nonzero.mean():.2f}%")
print(f"  TRAIN fake/real bright ratio: {train_ratios.mean():.3f}")
print(f"  TEST  fake/real bright ratio: {test_bright_ratios.mean():.3f}")
if train_nonzero.mean() > test_nonzero.mean() * 1.5:
    print(f"  >> TRAIN is {train_nonzero.mean()/test_nonzero.mean():.1f}x brighter than TEST")
    print(f"  >> This confirms the quality is DIFFERENT between train and test!")
else:
    print(f"  >> Quality is similar between train and test")
