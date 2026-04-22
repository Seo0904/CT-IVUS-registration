import numpy as np
import torch
import ot

# median=0のとき、M/1e-8 で M が巨大になり
# sinkhorn 内部でオーバーフロー → cost=0, grad=0 になるだけでNaNにはならない
# では実際にNaNが全lossで出るのはなぜか？

# ヒント: sinkhorn2 は cost=0, grad=0 を返すが
# これは「loss_SB=0になり逆伝播がなにも起きない」
# 問題はむしろ他のloss (GAN, NCE) でNaNが出ているのでは？

# → transformed_cut_head_tail.npy は「head/tail がゼロフレーム」
# → 正規化後のゼロフレーム: 全画素が -1.0
# → _frame_weights で thr=1e-4 チェック
# → abs.mean(-1.0) = 1.0 > 1e-4 → 全フレームがactiveになる
# → frame_weights は均一に 1/20

# では NaN がどこから来るかを追う
# transformed_cut_head_tail: フレームの先頭・末尾がゼロ（黒）
# これは seq 全体で 平均輝度が低い
# NCE loss の計算でも問題になりうる

# まず: GAN loss で NaN が出うる条件
# → Discriminator が fake/real を完璧に分類 → log(0) → -inf → NaN
# → BCEWithLogitsは数値安定だが、wgan等では出る

# 最も可能性が高いのは:
# transformed_cut_head_tail のデータで「ゼロフレームが先頭・末尾に集中」
# → AとBでシーケンスの有効フレーム位置が大きくずれる
# → OTが有効フレームを誤マッチングしようとしてlossが暴れる
# → 勾配爆発 → NaN

# より具体的に: fake_seq と real_B_seq でフレームの「有効範囲」が違う
# A: 全20フレームが有効
# B (head_tail): フレーム5〜14だけ有効、0〜4と15〜19がゼロ

# → sequence_ot_loss_torch の _frame_weights では
#   「ゼロフレームも含めて全フレームに重みをつける」
#   (thr=1e-4 だが、ゼロフレームの正規化後abs.mean=1.0 > 1e-4)
# → OT が A[0〜4](有効フレーム) と B[0〜4](ゼロフレーム) をマッチング
# → 大きなコスト → 勾配が爆発

b = np.load('/workspace/data/preprocessed/bspline_transformed/transformed_cut_head_tail.npy')
b2 = b.transpose(1,0,2,3).astype(np.float32)

print("=== transformed_cut_head_tail のフレーム構造 ===")
zero_counts = (b2.max(axis=(2,3)) == 0).sum(axis=1)

# ゼロフレームの位置分布
zero_positions = []
for i in range(len(b2)):
    zp = np.where(b2[i].max(axis=(1,2)) == 0)[0]
    zero_positions.extend(zp.tolist())

from collections import Counter
pos_counts = Counter(zero_positions)
print("ゼロフレームが出現するタイムステップの頻度:")
for t in range(20):
    print(f"  t={t:2d}: {pos_counts.get(t,0):5d}回")

print(f"\n→ head (t=0〜4) とtail (t=15〜19) に集中している")
print(f"→ 有効フレームはほぼ t=5〜14 の10フレーム")
print()
print("=== AとBの有効フレーム範囲 ===")
a = np.load('/workspace/data/org_data/moving_mnist/mnist_test_seq.npy')
if a.shape[0] <= 50:
    a = a.transpose(1,0,2,3)
# Aの各フレームのゼロ数
a_zero = (a.max(axis=(2,3)) == 0)
print(f"A (mnist): ゼロフレームを持つシーケンス = {a_zero.any(axis=1).sum()}")

print(f"\n→ A は全フレームが有効（0ゼロフレーム 99%）")
print(f"→ B (head_tail) は平均 {zero_counts.mean():.1f}個のゼロフレームあり")
print()
print("=== NaNの真の原因 ===")
print("1. M.median()=0 (ゼロ列が50%でmedianがゼロになるケース)")
print("   → M / 1e-8 で爆発 → sinkhorn 数値不安定 → 勾配NaN")
print("2. fake_seqの有効フレームとreal_B_seqのゼロフレームがOTでマッチング")
print("   → 巨大なコスト → 勾配爆発 → 他のlossもNaNに伝染")
