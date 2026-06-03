"""
best epoch モデルで train データを生成し、FID で使うのと同じ特徴空間
（InceptionV3 pool3, 2048 次元）で生成画像 fake_B を 2D 可視化するスクリプト。

各生成フレームを「ペアとなる正解画像 real_B があるか」で色分けする:
  - 青 (blue) : real_B フレームが有効（非ゼロ）＝ペアあり（FID に使われるフレーム）
  - 赤 (red)  : real_B フレームがゼロ＝ペアなし（cloze 等で正解が存在しない）

ペア判定は train.py / run_fid と同じ get_valid_frame_idx の基準
（正規化後 max > -1 + 1e-3）を 1 フレームごとに適用する。

次元削減は PCA と t-SNE の両方を出力する。

使用方法（オプションは run_test_*.sh と同じものを渡す。--phase train 推奨）:
    python3 visualize_fid_feature_space_best.py \
        --dataroot ... --dataroot_B ... --name ... --model wpsb --mode sb \
        --epoch best --phase train --num_test 200 ...
"""
import os

import numpy as np
import torch
import torchvision.utils as vutils
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.sequence_ot import get_valid_frame_idx

from vgg_sb.evaluations.inception import InceptionV3
from vgg_sb.evaluations.fid_score import get_activations


def extract_fake_frames(opt, save_dir, real_save_dir):
    """best epoch モデルで train データを生成し、fake_B フレームを PNG 保存。
    あわせて、ペアが存在する（有効な）real_B フレームも PNG 保存する。

    FID（run_fid）と同じ vutils.save_image(normalize=True) で保存することで、
    実際の FID と同一の特徴空間になるようにする。

    Returns:
        files      : list[str]   fake_B PNG パス（順序保持）
        labels     : np.ndarray (N,) bool  True=ペアあり, False=ペアなし
        real_files : list[str]   有効 real_B PNG パス（ペアありフレームのみ）
    """
    dataset = create_dataset(opt)
    print(f"Dataset size (phase={opt.phase}): {len(dataset)}")

    model = create_model(opt)

    files = []
    labels = []
    real_files = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc="Generating")):
            if i >= opt.num_test:
                break

            # 1 イテレーション目でネット初期化・best チェックポイント読み込み
            if i == 0:
                model.data_dependent_initialize(data, data)
                model.setup(opt)
                model.parallelize()
                model.eval()

            model.set_input(data)
            model.forward()

            fake_B = model.fake_B.detach().cpu()   # (B,T,C,H,W)
            real_B = model.real_B.detach().cpu()

            if real_B.dim() == 5:
                b, t, c, h, w = real_B.shape
                fake_flat = fake_B.view(b * t, c, h, w)
                real_flat = real_B.view(b * t, c, h, w)
            else:
                fake_flat = fake_B
                real_flat = real_B

            # フレームごとのペア有無（run_fid の get_valid_frame_idx と同じ基準）
            valid_idx = get_valid_frame_idx(real_flat)            # 有効フレーム index
            is_pair = torch.zeros(real_flat.shape[0], dtype=torch.bool)
            is_pair[valid_idx] = True

            for b in range(fake_flat.size(0)):
                fpath = os.path.join(save_dir, f"{i:04d}_{b:02d}.png")
                vutils.save_image(fake_flat[b], fpath, normalize=True)
                files.append(fpath)
                labels.append(bool(is_pair[b]))

                # ペアあり（有効）な real_B だけ保存（穴フレームのゼロ画像は除外）
                if is_pair[b]:
                    rpath = os.path.join(real_save_dir, f"{i:04d}_{b:02d}.png")
                    vutils.save_image(real_flat[b], rpath, normalize=True)
                    real_files.append(rpath)

    return files, np.array(labels, dtype=bool), real_files


def compute_inception_features(files, dims=2048):
    """FID と同じ InceptionV3 pool3 特徴 (N, dims) を計算。"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    act = get_activations(files, model, batch_size=32, dims=dims, cuda=cuda)
    return act


def find_overlap_idx(emb, labels, k=10, thr=0.5):
    """2D 埋め込み上で、近傍 k 個のうち反対クラスが thr 以上を占める点
    （＝相手クラスの領域に入り込んで重なっている点）の index を返す。"""
    from sklearn.neighbors import NearestNeighbors

    k_eff = min(k, len(emb) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(emb)
    _, idx = nn.kneighbors(emb)            # 先頭は自分自身
    neigh = idx[:, 1:]                      # 自分を除く
    opp_frac = (labels[neigh] != labels[:, None]).mean(axis=1)
    return np.where(opp_frac >= thr)[0]


def save_montage(files, sel_idx, labels, out_path, cols=20, border=2):
    """sel_idx の fake_B 画像をグリッド状に並べ、真のクラスの色で縁取りして保存。
    青枠=ペアあり, 赤枠=ペアなし。"""
    if len(sel_idx) == 0:
        print(f"[Montage] 重なり点なし: {out_path} はスキップ")
        return
    # 赤(ペアなし)→青(ペアあり) の順に並べると見やすい
    sel_idx = sorted(sel_idx, key=lambda i: labels[i])

    tiles = []
    for i in sel_idx:
        img = Image.open(files[i]).convert("RGB")
        color = (0, 0, 255) if labels[i] else (255, 0, 0)  # 青=pair, 赤=no-pair
        w, h = img.size
        canvas = Image.new("RGB", (w + 2 * border, h + 2 * border), color)
        canvas.paste(img, (border, border))
        tiles.append(canvas)

    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tw, rows * th), (30, 30, 30))
    for j, t in enumerate(tiles):
        r, c = divmod(j, cols)
        grid.paste(t, (c * tw, r * th))
    grid.save(out_path)
    n_red = int((~labels[sel_idx]).sum())
    n_blue = int(labels[np.array(sel_idx)].sum())
    print(f"Saved: {out_path}  (重なり {len(sel_idx)} 枚: 赤={n_red}, 青={n_blue})")


def reduce_and_plot(features, labels, files, out_dir, name, epoch,
                    k_overlap=10, thr_overlap=0.5):
    """PCA と t-SNE の 2D 散布図を保存（青=ペアあり, 赤=ペアなし）。
    さらに各手法で重なり点を検出し、散布図上で強調＋fake_B 画像モンタージュを出力。"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_pair = int(labels.sum())
    n_nopair = int((~labels).sum())
    print(f"[Features] total={len(labels)}  pair(blue)={n_pair}  no-pair(red)={n_nopair}")

    methods = {"pca": PCA(n_components=2, random_state=0)}
    # t-SNE は perplexity < n_samples 必須
    perplexity = min(30, max(5, len(features) // 4))
    methods["tsne"] = TSNE(n_components=2, random_state=0, perplexity=perplexity, init="pca")

    for key, reducer in methods.items():
        emb = reducer.fit_transform(features)
        overlap_idx = find_overlap_idx(emb, labels, k=k_overlap, thr=thr_overlap)
        ov_mask = np.zeros(len(labels), dtype=bool)
        ov_mask[overlap_idx] = True

        plt.figure(figsize=(8, 7))
        # 赤を先に、青を上に描く
        plt.scatter(emb[~labels, 0], emb[~labels, 1], s=10, c="red",
                    alpha=0.5, label=f"no pair (n={n_nopair})")
        plt.scatter(emb[labels, 0], emb[labels, 1], s=10, c="blue",
                    alpha=0.5, label=f"pair (n={n_pair})")
        # 重なり点を黒縁で強調
        plt.scatter(emb[ov_mask, 0], emb[ov_mask, 1], s=40, facecolors="none",
                    edgecolors="black", linewidths=0.8,
                    label=f"overlap (n={len(overlap_idx)})")
        plt.legend(loc="best")
        plt.title(f"FID feature space ({key.upper()}) - {name} epoch={epoch} (train, fake_B)")
        plt.xlabel(f"{key}-1")
        plt.ylabel(f"{key}-2")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"fid_featspace_{key}_train_{epoch}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

        # 重なり点の fake_B 画像モンタージュ
        montage_path = os.path.join(out_dir, f"overlap_montage_{key}_train_{epoch}.png")
        save_montage(files, overlap_idx, labels, montage_path)

    # 特徴とラベルも保存しておく（再描画・別手法用）
    np.savez(os.path.join(out_dir, f"fid_features_train_{epoch}.npz"),
             features=features, labels=labels)
    print(f"Saved: {os.path.join(out_dir, f'fid_features_train_{epoch}.npz')}")


def plot_with_realB(fake_features, fake_labels, real_features, out_dir, name, epoch):
    """fake_B（青=ペアあり / 赤=ペアなし）に加え、有効 real_B を緑で重ねた
    別画像を PCA・t-SNE それぞれで保存する。
    3 群を同じ 2D 空間に置くため、結合した特徴で次元削減を fit する。"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_fake = len(fake_features)
    all_feats = np.vstack([fake_features, real_features])  # 前半=fake, 後半=real_B

    n_pair = int(fake_labels.sum())
    n_nopair = int((~fake_labels).sum())
    n_real = len(real_features)
    print(f"[with realB] fake pair(blue)={n_pair}  fake no-pair(red)={n_nopair}  real_B(green)={n_real}")

    methods = {"pca": PCA(n_components=2, random_state=0)}
    perplexity = min(30, max(5, len(all_feats) // 4))
    methods["tsne"] = TSNE(n_components=2, random_state=0, perplexity=perplexity, init="pca")

    for key, reducer in methods.items():
        emb_all = reducer.fit_transform(all_feats)
        emb_fake = emb_all[:n_fake]
        emb_real = emb_all[n_fake:]

        plt.figure(figsize=(8, 7))
        plt.scatter(emb_fake[~fake_labels, 0], emb_fake[~fake_labels, 1], s=10,
                    c="red", alpha=0.5, label=f"fake_B no pair (n={n_nopair})")
        plt.scatter(emb_fake[fake_labels, 0], emb_fake[fake_labels, 1], s=10,
                    c="blue", alpha=0.5, label=f"fake_B pair (n={n_pair})")
        plt.scatter(emb_real[:, 0], emb_real[:, 1], s=10,
                    c="green", alpha=0.5, label=f"real_B (n={n_real})")
        plt.legend(loc="best")
        plt.title(f"FID feature space ({key.upper()}) - {name} epoch={epoch} (train, fake_B + real_B)")
        plt.xlabel(f"{key}-1")
        plt.ylabel(f"{key}-2")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"fid_featspace_{key}_train_{epoch}_with_realB.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    opt = TestOptions().parse()

    # train データの本当のペアで評価するため、ランダムペアリングを無効化
    opt.unsb = False
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    if not hasattr(opt, "phase") or opt.phase == "test":
        opt.phase = "train"

    out_dir = os.path.join(opt.results_dir, opt.name, f"fid_featspace_{opt.phase}_{opt.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    # モンタージュ用に fake_B / real_B の PNG は永続保存する（埋め込みの index と対応）
    frames_dir = os.path.join(out_dir, "frames")
    real_frames_dir = os.path.join(out_dir, "frames_realB")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(real_frames_dir, exist_ok=True)

    files, labels, real_files = extract_fake_frames(opt, frames_dir, real_frames_dir)
    features = compute_inception_features(files, dims=2048)
    reduce_and_plot(features, labels, files, out_dir, opt.name, opt.epoch)

    # real_B を緑で重ねた別画像も出力
    real_features = compute_inception_features(real_files, dims=2048)
    plot_with_realB(features, labels, real_features, out_dir, opt.name, opt.epoch)

    print(f"\nDone. Outputs in: {out_dir}")
