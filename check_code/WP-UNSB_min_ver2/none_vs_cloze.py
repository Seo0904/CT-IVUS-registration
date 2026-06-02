#!/usr/bin/env python3
"""
none vs cloze : 最適輸送（Sinkhorn）比較スクリプト

【データ】
  fake = transformed_global.npy        (N, T, H, W) in [0, 1] → [-1, 1] に変換
  tgt  = transformed_cloze_global.npy  (N, T, H, W) in [0, 1] → [-1, 1] に変換

  学習コード (moving_mnist_paired_dataset.py) の前処理に合わせて
  ToTensor → Normalize((0.5,),(0.5,)) と等価な変換 (x*2-1) を適用する。

【OT の設計 (sequence_ot.py の gen vs tgt をまねる)】
  各サンプル i に対して:
    fake_frames : (T, H, W)          — 全フレーム使用
    tgt_frames  : (N_valid_i, H, W)  — 黒フレーム (max == -1.0) を除いた有効フレームのみ
    コスト行列 M : (T, N_valid_i)    — squared L2
    輸送行列   P : (T, N_valid_i)    — Sinkhorn で算出

  ※ 学習コード sequence_ot.py と同じく [-1, 1] 正規化済みデータを使い、
     閾値 max > -1.0 + 1e-3 で黒フレームを除外する。
  ※ OT コストも学習コードと同じく <P, M> - reg * H(P) で計算する。

【保存物】
  OUT_DIR/
    P_matrices.npy    — object array (N,), 各要素は (T, N_valid_i) の輸送行列
    M_matrices.npy    — object array (N,), 各要素は (T, N_valid_i) の距離行列 (正規化済み)
    ot_costs.npy      — (N,) float64, 各サンプルの OT コスト <P,M> - reg*H(P)
    valid_counts.npy  — (N,) int32, 各サンプルの有効フレーム数
    valid_idx.npy     — object array (N,), 各サンプルの有効フレームインデックス
"""

import os
import numpy as np
import torch
import ot
import matplotlib.pyplot as plt

# -------------------------------------------------------
# パス設定
# -------------------------------------------------------
FAKE_PATH = "/workspace/data/preprocessed/bspline_transformed/transformed_global.npy"
TGT_PATH  = "/workspace/data/preprocessed/bspline_transformed/transformed_cloze_global.npy"
OUT_DIR   = "/workspace/check_code/WP-UNSB_min_ver2/ot_results_none_vs_cloze"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# ハイパーパラメータ (sequence_ot.py のデフォルトに合わせる)
# -------------------------------------------------------
REG         = 0.03      # Sinkhorn 正則化係数 (学習スクリプトの --lmda に合わせる)
ITERS       = 50        # Sinkhorn 反復回数 (学習コードの seq_ot_iters デフォルトに合わせる)
NORMALIZE   = "none"    # コスト行列のスケール正規化: "mean" / "median" / "max" / None
SINKHORN_METHOD = "sinkhorn_log"  # "sinkhorn" or "sinkhorn_log"

# -------------------------------------------------------
# 対象サンプル設定
#   None  → 全サンプル (N=10000) を処理
#   int   → そのインデックス1件だけ処理
#   list  → 指定インデックスのみ処理
#
# 学習コードの save_ot_snapshots では train/val 先頭 3件 (i=0,1,2) を処理する。
# 「train の 2系列目」= dataset index 1 (0-origin) に対応。
# -------------------------------------------------------
TARGET_IDX = 1   # 例: 1 → 2系列目のみ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# -------------------------------------------------------
# 可視化ヘルパー  (visualize_ot_details.py と同じ実装)
# -------------------------------------------------------
def save_heatmap(array, path, title=None, xlabel=None, ylabel=None, cmap="viridis"):
    """2D 配列をヒートマップとして保存する。"""
    array = np.asarray(array)
    if array.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(array, aspect="auto", origin="lower", cmap=cmap)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_curve(array, path, title=None, xlabel=None, ylabel=None):
    """1D 配列を折れ線グラフとして保存する。"""
    array = np.asarray(array).reshape(-1)
    if array.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(array, marker="o" if array.size <= 50 else None)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_U(P_np: np.ndarray) -> np.ndarray:
    """
    sequence_ot.py の monotone ブロックと同じ計算。
    P_np : (T, N_valid)
    → U  : (T,)  各 fake フレームが tgt のどのインデックスに対応するかの重心
    """
    T, N_valid = P_np.shape
    j_idx   = np.arange(N_valid, dtype=np.float64).reshape(1, N_valid)   # (1, N_valid)
    row_mass = P_np.sum(axis=1, keepdims=True) + 1e-8                     # (T, 1)
    U = np.sum(P_np * j_idx, axis=1) / row_mass.squeeze(1)               # (T,)
    return U


def save_seq_image(seq_arr, path, title):
    """
    フレーム系列を横一列に並べた画像として保存する。
    学習コードの wpsb_model.py: _save_seq_image と同じ実装。

    seq_arr : (T, H, W) または (T, C, H, W)、値は [-1, 1]
              学習コードの wpsb_model.py: _save_seq_image と同じく
              [-1, 1] → [0, 1] に変換してから描画する。
    """
    if seq_arr is None:
        return
    arr = np.array(seq_arr, dtype=np.float32)

    if arr.ndim == 3:          # (T, H, W) → (T, 1, H, W)
        T, H, W = arr.shape
        C = 1
        arr = arr[:, np.newaxis, ...]
    elif arr.ndim == 4:        # (T, C, H, W)
        T, C, H, W = arr.shape
    else:
        return

    # [-1, 1] → [0, 1] (学習コードの _save_seq_image と同じ)
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)

    # 横一列にフレームを並べたキャンバスを作成
    if C == 1:
        canvas = np.zeros((H, W * T), dtype=np.float32)
        for t in range(T):
            canvas[:, t * W:(t + 1) * W] = arr[t, 0]
        fig, ax = plt.subplots(figsize=(1.5 * T, 3))
        ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        canvas = np.zeros((H, W * T, C), dtype=np.float32)
        for t in range(T):
            canvas[:, t * W:(t + 1) * W, :] = arr[t].transpose(1, 2, 0)
        fig, ax = plt.subplots(figsize=(1.5 * T, 3))
        ax.imshow(canvas, vmin=0.0, vmax=1.0)

    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_ot(P_np, M_np, valid_idx, out_prefix,
                 fake_seq=None, tgt_seq=None):
    """
    P, M のヒートマップ・U の折れ線グラフ・フレーム系列画像を保存する。
    学習コードの _save_ot_details / visualize_ot_details.py と同じスタイル。

    out_prefix : 保存先のプレフィックス（例: /path/to/idx1）
    保存ファイル:
      {out_prefix}_M.png       — OT cost matrix  (T × N_valid)
      {out_prefix}_P.png       — OT transport plan (T × N_valid)
      {out_prefix}_U.png       — barycentric assignment curve (length T)
      {out_prefix}_fake_B.png  — fake (none) 全フレーム横並び   ← fake_seq 指定時
      {out_prefix}_real_B.png  — tgt (cloze) 全フレーム横並び  ← tgt_seq 指定時
    """
    # ── M ヒートマップ ────────────────────────────────────────────────────
    save_heatmap(
        M_np,
        path    = out_prefix + "_M.png",
        title   = "OT cost matrix M",
        xlabel  = "valid tgt index",
        ylabel  = "fake frame index",
        cmap    = "viridis",
    )

    # ── P ヒートマップ ────────────────────────────────────────────────────
    save_heatmap(
        P_np,
        path    = out_prefix + "_P.png",
        title   = "OT transport plan P",
        xlabel  = "valid tgt index",
        ylabel  = "fake frame index",
        cmap    = "viridis",
    )

    # ── U 折れ線 ─────────────────────────────────────────────────────────
    U = compute_U(P_np)
    save_curve(
        U,
        path    = out_prefix + "_U.png",
        title   = "OT barycentric assignments U",
        xlabel  = "fake frame index",
        ylabel  = "tgt barycentric idx",
    )

    # ── フレーム系列画像 (学習コードの _save_seq_image と同じ) ────────────
    if fake_seq is not None:
        save_seq_image(fake_seq, out_prefix + "_fake_B.png", "fake_B sequence (none)")
    if tgt_seq is not None:
        save_seq_image(tgt_seq,  out_prefix + "_real_B.png", "real_B sequence (cloze)")

    saved = ["_M.png", "_P.png", "_U.png"]
    if fake_seq is not None: saved.append("_fake_B.png")
    if tgt_seq  is not None: saved.append("_real_B.png")
    print(f"  → saved: {out_prefix}" + " / ".join(saved))
    print(f"    U = {np.round(U, 2)}")


# -------------------------------------------------------
# 有効フレーム検出
# sequence_ot.py の get_valid_frame_idx と同じ実装
#   [-1, 1] 範囲: max > -1.0 + 1e-3
# -------------------------------------------------------
def get_valid_frame_idx_01(seq_np: np.ndarray) -> np.ndarray:
    """
    seq_np : (T, H, W) in [-1, 1]
    黒フレーム (max <= -1.0) を除いた有効フレームインデックスを返す。
    全フレームが黒の場合は全インデックスを返す（フォールバック）。
    学習コード sequence_ot.py の get_valid_frame_idx と同じ閾値。
    """
    frame_max = seq_np.reshape(seq_np.shape[0], -1).max(axis=1)  # (T,)
    valid = np.where(frame_max > -1.0 + 1e-3)[0]
    if len(valid) == 0:
        valid = np.arange(seq_np.shape[0])
    return valid


# -------------------------------------------------------
# データ読み込み
# -------------------------------------------------------
print("Loading data...")
fake_all = np.load(FAKE_PATH).astype(np.float32)   # (N, T, H, W)
tgt_all  = np.load(TGT_PATH).astype(np.float32)    # (N, T, H, W)

N, T, H, W = fake_all.shape
print(f"  fake (raw): {fake_all.shape}  [{fake_all.min():.3f}, {fake_all.max():.3f}]")
print(f"  tgt  (raw): {tgt_all.shape}   [{tgt_all.min():.3f}, {tgt_all.max():.3f}]")

# 学習コードの前処理に合わせて [0,1] → [-1,1] に変換
# (moving_mnist_paired_dataset.py: ToTensor → Normalize((0.5,),(0.5,)) と等価)
if fake_all.max() > 1.0:
    fake_all = fake_all / 255.0
if tgt_all.max() > 1.0:
    tgt_all  = tgt_all  / 255.0
fake_all = fake_all * 2.0 - 1.0
tgt_all  = tgt_all  * 2.0 - 1.0

print(f"  fake ([-1,1]): [{fake_all.min():.3f}, {fake_all.max():.3f}]")
print(f"  tgt  ([-1,1]): [{tgt_all.min():.3f},  {tgt_all.max():.3f}]")
print(f"  N={N}, T={T}, H={H}, W={W}")


# -------------------------------------------------------
# 対象インデックスの解決
# -------------------------------------------------------
if TARGET_IDX is None:
    target_indices = list(range(N))
elif isinstance(TARGET_IDX, int):
    assert 0 <= TARGET_IDX < N, f"TARGET_IDX={TARGET_IDX} is out of range [0, {N})"
    target_indices = [TARGET_IDX]
else:
    target_indices = list(TARGET_IDX)

print(f"\nTarget indices: {target_indices}  ({len(target_indices)} sample(s))")


# -------------------------------------------------------
# サンプルごとに OT 計算
# -------------------------------------------------------
P_list         = []    # list of (idx, (T, N_valid_i) ndarray)
M_list         = []    # list of (idx, (T, N_valid_i) ndarray)
ot_costs_dict  = {}    # {sample_idx: float}
valid_counts   = {}    # {sample_idx: int}
valid_idx_list = {}    # {sample_idx: ndarray}
fake_seq_list  = {}    # {sample_idx: (T, H, W) ndarray}
tgt_seq_list   = {}    # {sample_idx: (T, H, W) ndarray}

print(f"Computing OT  (reg={REG}, iters={ITERS}, normalize={NORMALIZE})...")

for i in target_indices:
    fake_seq = fake_all[i]   # (T, H, W)
    tgt_seq  = tgt_all[i]    # (T, H, W)

    # ── tgt の有効フレームだけ使う (黒フレーム除外) ──────────────────────
    valid_idx = get_valid_frame_idx_01(tgt_seq)   # (N_valid,)
    n_valid   = len(valid_idx)
    valid_counts[i]   = n_valid
    valid_idx_list[i] = valid_idx

    tgt_valid = tgt_seq[valid_idx]   # (N_valid, H, W)

    # ── フラット化 ────────────────────────────────────────────────────────
    # sequence_ot.py: fake_flat = fake_sub.reshape(T, -1)
    #                 tgt_flat  = tgt_sub.reshape(N, -1)
    fake_flat = fake_seq.reshape(T,       -1).astype(np.float64)   # (T,       H*W)
    tgt_flat  = tgt_valid.reshape(n_valid, -1).astype(np.float64)  # (N_valid, H*W)

    # ── コスト行列 M = squared L2  (T, N_valid) ──────────────────────────
    # sequence_ot.py: M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2
    fake_t = torch.from_numpy(fake_flat).to(DEVICE)
    tgt_t  = torch.from_numpy(tgt_flat).to(DEVICE)
    M_t    = torch.cdist(fake_t, tgt_t, p=2) ** 2    # (T, N_valid)

    # ── スケール正規化 ────────────────────────────────────────────────────
    # sequence_ot.py の normalize == "mean" と同じ処理
    if NORMALIZE == "mean":
        s = M_t.detach().mean()
        M_t = M_t / (s + 1e-8)
    elif NORMALIZE == "median":
        s = M_t.detach().median()
        if s < 1e-6:
            s = M_t.detach().mean()
        M_t = M_t / (s + 1e-8)
    elif NORMALIZE == "max":
        s = M_t.detach().max()
        M_t = M_t / (s + 1e-8)
    elif NORMALIZE == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalize: {NORMALIZE}")

    M_np = M_t.cpu().numpy()   # (T, N_valid)

    # ── 一様分布 ──────────────────────────────────────────────────────────
    # sequence_ot.py: a = ones(T)/T,  b = ones(N)/N
    a = np.ones(T,       dtype=np.float64) / T
    b = np.ones(n_valid, dtype=np.float64) / n_valid

    # ── Sinkhorn 輸送計画 ─────────────────────────────────────────────────
    # sequence_ot.py: P = ot.sinkhorn(a, b, M, reg=reg, numItermax=iters, method=sinkhorn_type)
    P_np = ot.sinkhorn(
        a, b, M_np,
        reg=REG,
        numItermax=ITERS,
        method=SINKHORN_METHOD,
    )

    # ── OT コスト = <P, M> - reg * H(P) ─────────────────────────────────
    # 学習コード sequence_ot.py と同じ計算式:
    #   OT_entropy = -sum(P * log(P + 1e-8))
    #   ot_cost    = sum(P * M) - reg * OT_entropy
    OT_entropy  = -float(np.sum(P_np * np.log(P_np + 1e-8)))
    ot_cost_val = float(np.sum(P_np * M_np)) - REG * OT_entropy
    ot_costs_dict[i] = ot_cost_val

    P_list.append((i, P_np))
    M_list.append((i, M_np))

    # シーケンス画像も保存用に保持
    fake_seq_list[i] = fake_seq   # (T, H, W)
    tgt_seq_list[i]  = tgt_seq    # (T, H, W) ← 黒フレーム込みの元データ

    print(f"  [idx={i}]  valid_frames={n_valid:2d}  P.shape={P_np.shape}  ot_cost={ot_cost_val:.4f}")


# -------------------------------------------------------
# 保存
#   単一サンプルの場合: idx{i}_P.npy, idx{i}_M.npy, idx{i}_valid_idx.npy
#   複数サンプルの場合: object array として P_matrices.npy, M_matrices.npy
# -------------------------------------------------------
print("\nSaving results...")

if len(target_indices) == 1:
    # ── 1件のみ: そのインデックスを名前に含めてシンプルに保存 ──────────
    idx = target_indices[0]
    _, P_np = P_list[0]
    _, M_np = M_list[0]

    np.save(os.path.join(OUT_DIR, f"idx{idx}_P.npy"),         P_np)                # 輸送行列
    np.save(os.path.join(OUT_DIR, f"idx{idx}_M.npy"),         M_np)                # 距離行列
    np.save(os.path.join(OUT_DIR, f"idx{idx}_valid_idx.npy"), valid_idx_list[idx]) # 有効フレーム idx
    np.save(os.path.join(OUT_DIR, f"idx{idx}_ot_cost.npy"),   np.array([ot_costs_dict[idx]]))

    print(f"\n{'='*50}")
    print(f"Saved to: {OUT_DIR}")
    print(f"  idx{idx}_P.npy         : {P_np.shape}  (T={T}, N_valid={valid_counts[idx]})")
    print(f"  idx{idx}_M.npy         : {M_np.shape}")
    print(f"  idx{idx}_valid_idx.npy : {valid_idx_list[idx]}")
    print(f"  idx{idx}_ot_cost.npy   : {ot_costs_dict[idx]:.4f}")
    print(f"{'='*50}")

    # ── 可視化 ─────────────────────────────────────────────────────────
    print("\nVisualizing...")
    visualize_ot(P_np, M_np, valid_idx_list[idx],
                 out_prefix=os.path.join(OUT_DIR, f"idx{idx}"),
                 fake_seq=fake_seq_list[idx],
                 tgt_seq=tgt_seq_list[idx])

else:
    # ── 複数件: object array として保存 ──────────────────────────────────
    n_done = len(target_indices)
    P_arr  = np.empty(n_done, dtype=object)
    M_arr  = np.empty(n_done, dtype=object)
    vi_arr = np.empty(n_done, dtype=object)
    idx_arr = np.array(target_indices, dtype=np.int32)
    ot_costs_arr = np.array([ot_costs_dict[i] for i in target_indices], dtype=np.float64)

    for k, (i, P_np) in enumerate(P_list):
        P_arr[k]  = P_np
    for k, (i, M_np) in enumerate(M_list):
        M_arr[k]  = M_np
    for k, i in enumerate(target_indices):
        vi_arr[k] = valid_idx_list[i]

    np.save(os.path.join(OUT_DIR, "P_matrices.npy"),   P_arr)
    np.save(os.path.join(OUT_DIR, "M_matrices.npy"),   M_arr)
    np.save(os.path.join(OUT_DIR, "ot_costs.npy"),     ot_costs_arr)
    np.save(os.path.join(OUT_DIR, "valid_idx.npy"),    vi_arr)
    np.save(os.path.join(OUT_DIR, "target_indices.npy"), idx_arr)

    vc_arr = np.array([valid_counts[i] for i in target_indices], dtype=np.int32)

    print(f"\n{'='*50}")
    print(f"Saved to: {OUT_DIR}")
    print(f"  P_matrices.npy     : object array ({n_done},)")
    print(f"  M_matrices.npy     : object array ({n_done},)")
    print(f"  ot_costs.npy       : {ot_costs_arr}")
    print(f"  target_indices.npy : {idx_arr}")
    print(f"{'='*50}")
    print(f"OT cost  mean={ot_costs_arr.mean():.4f}  std={ot_costs_arr.std():.4f}")
    print(f"Valid frames  mean={vc_arr.mean():.1f}  min={vc_arr.min()}  max={vc_arr.max()}")

    # ── 可視化 (各サンプルごと) ────────────────────────────────────────
    print("\nVisualizing...")
    for k, (i_idx, P_np_k) in enumerate(P_list):
        _, M_np_k = M_list[k]
        visualize_ot(P_np_k, M_np_k, valid_idx_list[i_idx],
                     out_prefix=os.path.join(OUT_DIR, f"idx{i_idx}"),
                     fake_seq=fake_seq_list[i_idx],
                     tgt_seq=tgt_seq_list[i_idx])
