"""
特徴空間と輸送行列の可視化スクリプト（WP-UNSB / UNSB 共通）

best checkpoint をロードし、test セットを再推論して以下を保存する:

  1. 特徴空間（FID と同じ InceptionV3 pool3, 2048 次元）
       生成系列フレーム vs ターゲット系列フレームを色分けして 2D 散布図に。
       PCA と t-SNE の両方を出力。既定で 1000 系列。
       使う生成出力は evaluate_all.py と同じ最終 NFE 出力 fake_{num_timesteps}。

  2. 輸送行列（sequence OT plan P）
       先頭 数系列ぶんの P をヒートマップ + .npy で保存。
       GT 対角（生成フレーム i ↔ ターゲット原フレーム i）を重ねて表示。
       穴あき(cloze)ターゲット（--ot_match_file_B）に対して計算する: N≠M を作って
       初めて「系列長が違っても対応を回復し余剰フレームを捨てる」挙動が見える。
       モデルが UOT 学習（seq_ot_unbalanced=ρ）なら UOT 版と balanced(OT) 版を
       並べて出力し、「balanced は余剰も無理に輸送 / UOT は捨てる」を対比する。

オプションは run_test_*.sh と同じものを渡せる（TestOptions 流用）。追加オプション:
  --feat_num_seq N      特徴空間に使う系列数（default 1000）
  --ot_num_seq N        輸送行列を保存する先頭系列数（default 3）
  --ot_match_file_B F   輸送行列用の穴あき(cloze)ターゲット npy（matching と同じ）
  --tsne_subsample N    t-SNE に渡す総点数の上限（0=全点。default 10000）
  --viz_out DIR         保存先（default: results_dir/name/feature_ot_{phase}_{epoch}）
"""
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

from vgg_sb.evaluations.inception import InceptionV3
from vgg_sb.evaluations.fid_score import get_activations

import inspect
try:
    # モデルが実際に使うディスパッチャ（solver=geo, unbalanced=ρ を含む）。
    # これでモデルが balanced OT か UOT かを忠実に再現できる。
    from models.sequence_ot import sequence_ot_loss as ot_dispatch
except Exception:
    ot_dispatch = None


def compute_plan(fake_seq, tgt_seq, opt, unbalanced):
    """モデルと同じ引数で sequence_ot_loss を呼び、(P, valid_idx) を返す。
    unbalanced=None で balanced OT、値（ρ）を渡すと UOT（geo ソルバのみ対応）。
    ver2 のように unbalanced/geo 引数を持たないディスパッチャでも壊れないよう
    シグネチャでフィルタする。"""
    kwargs = dict(
        solver=getattr(opt, 'seq_ot_solver', 'pot'),
        reg=getattr(opt, 'lmda', 0.05),
        iters=getattr(opt, 'seq_ot_iters', 50),
        monotone=getattr(opt, 'seq_ot_monotone', True),
        monotone_penalty=getattr(opt, 'seq_ot_monotone_penalty', 1.0),
        P_entropy=getattr(opt, 'seq_ot_p_entropy', False),
        P_entropy_penalty=getattr(opt, 'seq_ot_p_entropy_penalty', 1.0),
        ot_divergence=getattr(opt, 'seq_ot_divergence', False),
        ot_divergence_penalty=getattr(opt, 'seq_ot_divergence_penalty', -0.5),
        normalize=(None if getattr(opt, 'seq_ot_normalize', 'mean') == 'none'
                   else getattr(opt, 'seq_ot_normalize', 'mean')),
        sinkhorn_type=getattr(opt, 'sinkhorn_type', 'sinkhorn'),
        geo_scaling=getattr(opt, 'seq_ot_geo_scaling', 0.99),
        geo_p=getattr(opt, 'seq_ot_geo_p', 2),
        unbalanced=unbalanced,
        return_details=True,
    )
    sig = inspect.signature(ot_dispatch)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    _, details = ot_dispatch(fake_seq, tgt_seq, **kwargs)
    return details['P'], details['valid_idx']


def _pop_arg(argv, key, default):
    if key in argv:
        i = argv.index(key)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def valid_frame_idx(seq):
    """seq: (T,C,H,W) in [-1,1]。ゼロ画素フレーム(max==-1)を除いた index。"""
    frame_max = seq.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]
    valid = (frame_max > -1.0 + 1e-3).nonzero(as_tuple=False).squeeze(1)
    if valid.numel() == 0:
        valid = torch.arange(seq.shape[0], device=seq.device)
    return valid


def flatten_seq(x):
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        return x.view(b * t, c, h, w)
    return x


def build_seq_tensor(seq_np, device):
    """(T,H,W) -> (T,1,H,W) in [-1,1]（dataset の grayscale transform 同等）"""
    x = torch.from_numpy(np.asarray(seq_np)).float()
    if x.max() > 1:
        x = x / 255.0
    x = x * 2 - 1
    return x.unsqueeze(1).to(device)


def compute_inception_features(files, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    return get_activations(files, model, batch_size=32, dims=dims, cuda=cuda)


def scatter_2d(emb_gen, emb_tgt, key, name, epoch, out_path):
    plt.figure(figsize=(8, 7))
    plt.scatter(emb_tgt[:, 0], emb_tgt[:, 1], s=8, c="tab:blue",
                alpha=0.45, label=f"target (n={len(emb_tgt)})")
    plt.scatter(emb_gen[:, 0], emb_gen[:, 1], s=8, c="tab:red",
                alpha=0.45, label=f"generated (n={len(emb_gen)})")
    plt.legend(loc="best")
    plt.title(f"Feature space ({key.upper()}) - {name} epoch={epoch}")
    plt.xlabel(f"{key}-1")
    plt.ylabel(f"{key}-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_space(gen_feats, tgt_feats, out_dir, name, epoch, tsne_subsample):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_gen, n_tgt = len(gen_feats), len(tgt_feats)
    all_feats = np.vstack([gen_feats, tgt_feats])
    labels = np.concatenate([np.zeros(n_gen, dtype=int), np.ones(n_tgt, dtype=int)])  # 0=gen,1=tgt
    print(f"[feature space] generated={n_gen}  target={n_tgt}")

    # ---- PCA（全点）----
    pca = PCA(n_components=2, random_state=0)
    emb = pca.fit_transform(all_feats)
    scatter_2d(emb[labels == 0], emb[labels == 1], "pca", name, epoch,
               os.path.join(out_dir, f"featspace_pca_{epoch}.png"))

    # ---- t-SNE（高速化のため PCA-50 で前処理。必要なら層別サブサンプル）----
    idx = np.arange(len(all_feats))
    if tsne_subsample and len(all_feats) > tsne_subsample:
        rng = np.random.default_rng(0)
        per = tsne_subsample // 2
        gi = rng.choice(np.where(labels == 0)[0], min(per, n_gen), replace=False)
        ti = rng.choice(np.where(labels == 1)[0], min(per, n_tgt), replace=False)
        idx = np.concatenate([gi, ti])
        print(f"[t-SNE] subsampled to {len(idx)} points (cap={tsne_subsample})")
    sub = all_feats[idx]
    sub_labels = labels[idx]
    pca50 = PCA(n_components=min(50, sub.shape[1]), random_state=0).fit_transform(sub)
    perplexity = min(30, max(5, len(sub) // 4))
    emb_t = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                 init="pca").fit_transform(pca50)
    scatter_2d(emb_t[sub_labels == 0], emb_t[sub_labels == 1], "tsne", name, epoch,
               os.path.join(out_dir, f"featspace_tsne_{epoch}.png"))

    np.savez(os.path.join(out_dir, f"features_{epoch}.npz"),
             gen=gen_feats, tgt=tgt_feats)
    print(f"Saved: {os.path.join(out_dir, f'features_{epoch}.npz')}")


def save_transport_matrix(P, valid_idx, seq_id, out_dir, name, epoch, tag):
    """P:(T,N) のヒートマップ + .npy 保存。GT 対角を重ねる。
    tag: 'OT'(balanced) / 'UOT'(unbalanced) などファイル名・タイトルの識別子。"""
    P = P.detach().cpu().numpy()
    vidx = valid_idx.detach().cpu().numpy()  # 列 -> 原ターゲット index
    T, N = P.shape
    row_mass = P.sum(axis=1)

    np.save(os.path.join(out_dir, f"P_seq{seq_id}_{tag}_{epoch}.npy"), P)

    plt.figure(figsize=(max(5, N * 0.5), max(5, T * 0.35)))
    plt.imshow(P, aspect="auto", cmap="viridis")
    plt.colorbar(label="transport mass")
    # GT 対角: 列 k（原 index vidx[k]）の正解行は vidx[k]
    for k, orig in enumerate(vidx):
        if 0 <= orig < T:
            plt.scatter(k, orig, marker="o", s=60, facecolors="none",
                        edgecolors="red", linewidths=1.5)
    # argmax（手法が選んだ対応）
    pred = P.argmax(axis=1)
    plt.scatter(pred, np.arange(T), marker="x", s=40, c="white", linewidths=1.2)
    plt.xlabel("target frame (valid, original index)")
    plt.ylabel("generated frame")
    plt.xticks(range(N), [str(int(v)) for v in vidx], fontsize=7)
    plt.yticks(range(T), fontsize=7)
    plt.title(f"Transport matrix ({tag}) seq{seq_id} - {name} epoch={epoch}\n"
              f"(red o = GT diagonal, white x = argmax, transported mass={P.sum():.2f})")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"transport_seq{seq_id}_{tag}_{epoch}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}  (transported mass={P.sum():.3f}, "
          f"dropped rows[mass<0.2/T]={(row_mass < 0.2 / T).sum()}/{T})")


def main():
    argv = sys.argv
    feat_num_seq = int(_pop_arg(argv, '--feat_num_seq', 1000))
    ot_num_seq = int(_pop_arg(argv, '--ot_num_seq', 3))
    ot_match_file_B = _pop_arg(argv, '--ot_match_file_B', '')
    tsne_subsample = int(_pop_arg(argv, '--tsne_subsample', 10000))
    viz_out = _pop_arg(argv, '--viz_out', '')

    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.unsb = False

    device = torch.device(f'cuda:{opt.gpu_ids[0]}'
                          if (len(opt.gpu_ids) > 0 and torch.cuda.is_available()) else 'cpu')

    is_seq_model = (getattr(opt, 'model', '') == 'wpsb') and (ot_dispatch is not None)
    rho = getattr(opt, 'seq_ot_unbalanced', None)  # UOT の ρ（None なら balanced 学習）

    if not viz_out:
        viz_out = os.path.join(opt.results_dir, opt.name, f'feature_ot_{opt.phase}_{opt.epoch}')
    os.makedirs(viz_out, exist_ok=True)
    gen_dir = os.path.join(viz_out, 'frames_gen')
    tgt_dir = os.path.join(viz_out, 'frames_tgt')
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    # 輸送行列用の穴あきターゲット（任意。dataset と同じ split に切る）
    match_arr = None
    if is_seq_model and ot_match_file_B:
        path = ot_match_file_B if os.path.isabs(ot_match_file_B) else os.path.join(
            getattr(opt, 'dataroot_B', '') or opt.dataroot, ot_match_file_B)
        arr = np.load(path)
        if arr.shape[0] == 20:
            arr = np.transpose(arr, (1, 0, 2, 3))
        if opt.phase == 'train':
            match_arr = arr[:1000]
        elif opt.phase == 'val':
            match_arr = arr[1000:1200]
        else:
            match_arr = arr[1200:]
        print(f'[OT] holed target: {path} ({match_arr.shape[0]} seq, split {opt.phase})')

    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    model = create_model(opt)

    n_steps = int(getattr(opt, 'num_timesteps', 0))
    gen_files, tgt_files = [], []
    fake_source = 'fake_B'

    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(dataset, dataset2)):
            if i == 0:
                try:
                    model.data_dependent_initialize(data, data2)
                except TypeError:
                    model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                model.eval()
            if i >= feat_num_seq:
                break

            try:
                model.set_input(data, data2)
            except TypeError:
                model.set_input(data)
            model.test()

            if not (hasattr(model, 'fake_B') and hasattr(model, 'real_B')):
                continue

            fake_final = getattr(model, f'fake_{n_steps}', None)
            fake_source = f'fake_{n_steps}' if fake_final is not None else 'fake_B'
            fake = flatten_seq(fake_final if fake_final is not None else model.fake_B).to(device)
            real = flatten_seq(model.real_B).to(device)

            vidx = valid_frame_idx(real)
            fake_v = fake[vidx].cpu()
            real_v = real[vidx].cpu()

            # 特徴空間用に PNG 保存（FID と同じ normalize=True で保存）
            for b in range(fake_v.size(0)):
                gp = os.path.join(gen_dir, f"{i:05d}_{b:02d}.png")
                rp = os.path.join(tgt_dir, f"{i:05d}_{b:02d}.png")
                vutils.save_image(fake_v[b], gp, normalize=True)
                vutils.save_image(real_v[b], rp, normalize=True)
                gen_files.append(gp)
                tgt_files.append(rp)

            # 輸送行列（先頭 ot_num_seq 系列）
            #   - モデルが実際に使う設定（solver=geo, unbalanced=ρ）で計算 → UOT は余剰を捨てる
            #   - UOT 学習（ρ あり）なら、比較用に balanced(OT) 版も並べて出す
            #     （method.tex の「balanced は余剰も無理に輸送 / UOT は捨てる」の図）
            if is_seq_model and i < ot_num_seq:
                ot_real = real
                if match_arr is not None and i < match_arr.shape[0]:
                    ot_real = build_seq_tensor(match_arr[i], device)
                # モデル本来の設定（ρ があれば UOT、無ければ OT）
                primary_tag = 'UOT' if rho else 'OT'
                try:
                    P, vidx = compute_plan(fake, ot_real, opt, unbalanced=rho)
                    save_transport_matrix(P, vidx, i, viz_out, opt.name, opt.epoch, primary_tag)
                except Exception as e:
                    print(f'[OT] seq{i} {primary_tag} skip: {e}')
                # UOT 学習なら balanced 版も（同じ生成・同じターゲットで比較）
                if rho:
                    try:
                        P_b, vidx_b = compute_plan(fake, ot_real, opt, unbalanced=None)
                        save_transport_matrix(P_b, vidx_b, i, viz_out, opt.name, opt.epoch, 'OT')
                    except Exception as e:
                        print(f'[OT] seq{i} OT(balanced) skip: {e}')

            if i % 50 == 0:
                print(f'[{i}] collected {len(gen_files)} gen frames (output={fake_source})')

    if not gen_files:
        print('No frames collected.')
        return

    print(f'Extracting Inception features for {len(gen_files)} gen + {len(tgt_files)} tgt frames ...')
    gen_feats = compute_inception_features(gen_files)
    tgt_feats = compute_inception_features(tgt_files)
    plot_feature_space(gen_feats, tgt_feats, viz_out, opt.name, opt.epoch, tsne_subsample)

    print(f'\nDone. Outputs in: {viz_out}  (output={fake_source})')


if __name__ == '__main__':
    main()
