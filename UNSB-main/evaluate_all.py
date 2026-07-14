"""
統合定量評価スクリプト（WP-UNSB / UNSB 共通）

best checkpoint をロードして test セットを再推論し、以下をまとめて算出する:

  1. ペア指標（フレーム単位 GT、非穴フレームの対角で評価）
       L1 / L2 / PSNR / SSIM / LPIPS
  2. 分布指標（同一の test 系列で統一）
       FID / KID(mean ± std)（torch_fidelity, フレーム単位）と
       FVD（r3d_18, 系列=動画クリップ単位）を以下のペアで算出:
         gen vs tgt（メイン指標）
         gen vs src（ソースから離れたか = 入力コピーの検出）
         src vs tgt（ドメイン間距離 = 恒等写像の上限リファレンス）
         src vs src / tgt vs tgt（系列単位の半分割 2 群 = 床）
       ※ FVD は FID と同じ一時 PNG を系列単位で束ね直して計算。
         backend=i3d（既定）なら公式 FVD と同じ I3D(Kinetics-400) 特徴・前処理を使うため
         論文 FVD と比較可能な絶対値が出る（StyleGAN-V 配布の i3d_torchscript.pt が必要）。
         重みが無い / backend=r3d18 の場合は torchvision r3d_18 にフォールバックし、
         この場合は論文 FVD の絶対値とは比較不可で自分の実験どうしの相対比較用になる。
  3. matching accuracy（WP-UNSB 系のみ）
       学習済み生成器に対し sequence OT で推定したフレーム間対応が
       「生成フレーム i ↔ ターゲット原フレーム i」の対角 GT とどれだけ一致するか

オプションは test.py / run_test_*.sh とまったく同じものを渡せる
（TestOptions をそのまま流用）。追加オプション:
  --eval_kid_subset N   KID のサブセットサイズ上限（default 900）
  --eval_fvd 0|1        FVD(系列単位) を算出するか（default 1）
  --eval_fvd_backend B  FVD 特徴抽出器 i3d|r3d18（default i3d）。i3d は公式と同じで
                        論文比較可能、r3d18 は torchvision のみで動くが相対比較用。
  --eval_fvd_i3d PATH   i3d backend 用の i3d_torchscript.pt のパス
                        （default: 環境変数 FVD_I3D_PATH か
                         /workspace/.cache/fvd/i3d_torchscript.pt）。
                        見つからなければ自動で r3d18 にフォールバックする。
  --eval_fvd_viz 0|1    FVD 特徴空間の可視化と最近傍対応（検索）チェックを出すか（default 0）。
                        eval_out/fvd_viz/ に 2D 散布図(PCA/t-SNE)・最近傍を線で結んだ図・
                        マッチ例モンタージュ・json を保存し、gen 特徴の最近傍 tgt が
                        同一サンプル(同じ系列)になる割合(top1/top5)を測る。
  --eval_paired 0|1     ペア指標(SSIM/PSNR/L1/L2/LPIPS)を計算するか（default 1）。
                        0 にすると丸ごとスキップし result.paired=None / txt に skip 表記。
                        分布指標(FID/KID/FVD)・matching・可視化には影響しない。
  --eval_out DIR        結果 JSON/txt の保存先（default: results_dir/name/phase_epoch）
  --eval_txt_name NAME  可読テキストのファイル名（default evaluation_metrics.txt）。
                        拡張子 .txt は自動補完。JSON 名(evaluation_metrics.json)は固定。
  --eval_lpips_net NET  LPIPS バックボーン alex|vgg（default alex）
  --eval_match_file_B F matching accuracy 用の穴あき(cloze)ターゲット npy。
                        matching は「系列長が違っても(N≠M)対応を回復できるか」という
                        本手法の中核主張を測る指標なので、N≠M を作る穴あきが正しい。
                        画質指標(SSIM/PSNR/LPIPS/FID)は別主張(忠実な再現)なので
                        data_file_B の穴なしフル GT を使う。両者でターゲットが違う点に注意。
                        未指定なら matching も data_file_B(穴なし)になり、N≠M を一切
                        テストしない自明な対角(≈1.0)になるため非推奨。

使い方は run_evaluate_all.sh を参照。
"""
import os
import re
import sys
import json
import tempfile
import shutil
from math import log10

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util


# --------------------------------------------------------------------------
# 追加オプションを TestOptions に通す前に抜き取る
# --------------------------------------------------------------------------
def _pop_arg(argv, key, default):
    if key in argv:
        i = argv.index(key)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


# --------------------------------------------------------------------------
# 穴フレーム検出（models.sequence_ot.get_valid_frame_idx と同等。
# UNSB-main には sequence_ot が無いので自前で持つ）
# --------------------------------------------------------------------------
def valid_frame_idx(seq):
    """seq: (T, C, H, W) in [-1, 1]。ゼロ画素フレーム(max == -1)を除いた index。"""
    frame_max = seq.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]  # (T,)
    valid = (frame_max > -1.0 + 1e-3).nonzero(as_tuple=False).squeeze(1)
    if valid.numel() == 0:
        valid = torch.arange(seq.shape[0], device=seq.device)
    return valid


# --------------------------------------------------------------------------
# ペア指標（[-1,1] / [0,1] どちらの入力でも [0,1] に揃えて計算）
# --------------------------------------------------------------------------
def _to01(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.min() < 0:
        x = (x + 1) / 2
    return x.clamp(0, 1)


def _ssim(img1, img2, window_size=11):
    img1, img2 = _to01(img1), _to01(img2)
    channel = img1.size(1)
    gauss = torch.tensor(
        [np.exp(-(k - window_size // 2) ** 2 / (2 * 1.5 ** 2)) for k in range(window_size)]
    )
    gauss = (gauss / gauss.sum()).float()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous().to(img1.device).type_as(img1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def _psnr(img1, img2):
    img1, img2 = _to01(img1), _to01(img2)
    mse = F.mse_loss(img1, img2).item()
    return float('inf') if mse == 0 else 10 * log10(1.0 / mse)


def _l2(img1, img2):
    return F.mse_loss(_to01(img1), _to01(img2)).item()


def _l1(img1, img2):
    return F.l1_loss(_to01(img1), _to01(img2)).item()


# --------------------------------------------------------------------------
# (B,C,H,W) / (B,T,C,H,W) -> (N,C,H,W)
# --------------------------------------------------------------------------
def flatten_seq(x):
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        return x.view(b * t, c, h, w)
    return x


def save_frames(frames, out_dir, group, start):
    """frames: (N,C,H,W) in [-1,1] を 3ch PNG として書き出す。
       ファイル名は {group}_{通し番号:06d}.png。group（'_' より前）が系列 ID で、
       半分割はこの group 単位で行う。フレーム単位 dataset（UNSB-main）でも
       同一系列のフレームが同じ group にまとまるようにする。"""
    os.makedirs(out_dir, exist_ok=True)
    arr = (((frames.clamp(-1, 1) + 1) / 2) * 255).round().byte().cpu().numpy()  # (N,C,H,W)
    for k, img in enumerate(arr):  # (C,H,W)
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        img = np.transpose(img, (1, 2, 0))  # (H,W,3)
        Image.fromarray(img).save(os.path.join(out_dir, f'{group}_{start + k:06d}.png'))
    return start + arr.shape[0]


def seq_tag_of(data, fallback):
    """dataset の A_paths から系列 ID を取り出す。
       WP-UNSB: 'seq1234' / UNSB-main: 'seq1234_frame5' -> どちらも 'seq1234'。"""
    paths = data.get('A_paths', None)
    if paths is None:
        return fallback
    p = paths[0] if isinstance(paths, (list, tuple)) else paths
    tag = str(p).split('_')[0].replace('/', '')
    return tag if tag else fallback


def half_split_by_seq(src_dir, tmp_root, tag, seed=0):
    """src_dir 内の PNG を系列 prefix（'_' より前）単位でランダム半分割し、
       symlink で 2 つのディレクトリを作って返す。"""
    files = sorted(os.listdir(src_dir))
    prefixes = sorted({f.rsplit('_', 1)[0] for f in files})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(prefixes))
    half = len(prefixes) // 2
    g1 = {prefixes[i] for i in perm[:half]}
    d1 = os.path.join(tmp_root, f'{tag}_h1')
    d2 = os.path.join(tmp_root, f'{tag}_h2')
    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)
    for f in files:
        dst = d1 if f.rsplit('_', 1)[0] in g1 else d2
        os.symlink(os.path.abspath(os.path.join(src_dir, f)), os.path.join(dst, f))
    return d1, d2


# --------------------------------------------------------------------------
# FVD（系列=動画クリップ単位の FID）
#   FID は画像 1 枚を Inception 特徴空間に落とすが、FVD は系列(T,H,W)=動画クリップを
#   I3D(動画分類器)特徴空間に落として時間方向の一貫性まで含めて分布距離を測る。
#   距離計算(Fréchet 距離)は FID と同一。
#   特徴抽出器は torchvision r3d_18(Kinetics-400 学習済み, 512 次元, 凍結)。
#   ※ 論文 FVD の絶対値とは直接比較不可(バックボーンが違う)。自分の実験どうしの相対比較用。
#   FID/KID と同じ一時 PNG ディレクトリを系列(prefix='_'より前)単位で束ね直して使う。
# --------------------------------------------------------------------------
_EVAL_FRAME_RE = re.compile(r'(.+)_(\d+)\.(png|jpg|jpeg|bmp)$', re.IGNORECASE)


class FVDExtractor:
    """Kinetics-400 学習済み r3d_18 を fc 除去して 512 次元の固定特徴抽出器に。"""
    def __init__(self, device):
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.to(device).eval()
        t = weights.transforms()
        self.mean = torch.tensor(t.mean).view(1, 3, 1, 1, 1).to(device)
        self.std = torch.tensor(t.std).view(1, 3, 1, 1, 1).to(device)
        self.size = 112
        self.device = device

    @torch.no_grad()
    def feat(self, clip_u8):
        """clip_u8: (T,H,W) uint8 グレースケール -> (512,) numpy 特徴。"""
        x = torch.from_numpy(clip_u8).float().to(self.device) / 255.0  # (T,H,W)
        t, h, w = x.shape
        x = x.reshape(t, 1, h, w).repeat(1, 3, 1, 1)                   # (T,3,H,W) gray->3ch
        x = F.interpolate(x, size=(self.size, self.size),
                          mode='bilinear', align_corners=False)        # (T,3,112,112)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)                          # (1,3,T,112,112)
        x = (x - self.mean) / self.std
        return self.model(x).squeeze(0).cpu().numpy()                   # (512,)


class I3DExtractor:
    """公式 FVD と同じ I3D(Kinetics-400) を使う特徴抽出器。
       StyleGAN-V 等で配布されている TorchScript 版 I3D (i3d_torchscript.pt) を読み、
       /workspace/frechet_video_distance の DeepMind I3D と同一特徴・同一前処理を再現する。
       そのため出てくる FVD は論文 FVD と比較可能な絶対値になる（r3d_18 版と違い相対専用ではない）。
       重み i3d_torchscript.pt は別途用意してパス指定する（ライセンス上同梱しない）。"""
    def __init__(self, device, weights_path):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f'I3D weights not found: {weights_path}')
        # detector は (B,C,T,H,W) を受け、rescale で [0,255]->[-1,1]、resize で 224x224 にする
        self.model = torch.jit.load(weights_path, map_location='cpu').eval().to(device)
        self.device = device

    @torch.no_grad()
    def feat(self, clip_u8):
        """clip_u8: (T,H,W) uint8 グレースケール -> (D,) numpy 特徴（I3D 既定で D=400）。"""
        x = torch.from_numpy(clip_u8).float().to(self.device)           # (T,H,W) in [0,255]
        t, h, w = x.shape
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)                           # (T,3,H,W) gray->3ch
        x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()            # (1,3,T,H,W)
        # ↑ permute 後は非連続。TorchScript I3D 内部の view が連続性を要求し落ちるため contiguous 必須。
        # rescale=[0,255]->[-1,1], resize=224。公式 preprocess と同じ。
        feat = self.model(x, rescale=True, resize=True, return_features=True)
        return feat.squeeze(0).cpu().numpy()                            # (D,)


def load_clips_eval_grouped(folder):
    """evaluate_all が書いた {group}_{idx:06d}.png を系列(group)単位で束ねる。
       戻り値: (clips, groups)。clips=list[(T,H,W) uint8]、groups=list[str]（group ID, 昇順）。"""
    by_grp = {}
    for fn in os.listdir(folder):
        m = _EVAL_FRAME_RE.match(fn)
        if m is None:
            continue
        grp, idx = m.group(1), int(m.group(2))
        by_grp.setdefault(grp, []).append((idx, os.path.join(folder, fn)))
    clips, groups = [], []
    for grp in sorted(by_grp):
        frames = sorted(by_grp[grp], key=lambda x: x[0])
        imgs = [np.array(Image.open(fp).convert('L')) for _, fp in frames]
        clips.append(np.stack(imgs, axis=0))  # (T,H,W) 可変長可
        groups.append(grp)
    return clips, groups


def load_clips_eval(folder):
    """list[(T,H,W) uint8] だけ欲しいとき用の薄いラッパ。"""
    clips, _ = load_clips_eval_grouped(folder)
    return clips


def _frechet_distance(act1, act2, eps=1e-6):
    mu1, mu2 = act1.mean(0), act2.mean(0)
    sig1, sig2 = np.cov(act1, rowvar=False), np.cov(act2, rowvar=False)
    sig1, sig2 = np.atleast_2d(sig1), np.atleast_2d(sig2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + off).dot(sig2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2.0 * np.trace(covmean))


def compute_fvd(extractor, d1, d2, label):
    """d1, d2: PNG ディレクトリ。系列単位で r3d_18 特徴 -> Fréchet 距離。"""
    if extractor is None:
        return None
    try:
        c1, c2 = load_clips_eval(d1), load_clips_eval(d2)
        if len(c1) < 2 or len(c2) < 2:
            print(f'[fvd] {label}: skip (clips {len(c1)}/{len(c2)} < 2)')
            return None
        a1 = np.stack([extractor.feat(c) for c in c1], axis=0)
        a2 = np.stack([extractor.feat(c) for c in c2], axis=0)
        val = _frechet_distance(a1, a2)
        print(f'[fvd] {label}: FVD={val:.4f}  (n={len(c1)}/{len(c2)})')
        return {'FVD': val, 'n1': len(c1), 'n2': len(c2)}
    except Exception as e:
        print(f'[fvd] {label} failed: {e}')
        return None


# --------------------------------------------------------------------------
# FVD 特徴空間の可視化 + 最近傍対応（検索）チェック
#   gen と tgt は save_frames で同じ系列 group を共有するので、
#   gen clip i の最近傍 tgt が同じ group(=i) なら「特徴空間で同一サンプルに対応」と判定。
#   出力: 2D 散布図(PCA/t-SNE), 最近傍を線でつないだ図, マッチ例モンタージュ, json。
# --------------------------------------------------------------------------
def fvd_feature_report(extractor, gen_dir, tgt_dir, out_dir, label='gen_vs_tgt'):
    if extractor is None:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.spatial.distance import cdist
        from sklearn.decomposition import PCA
    except Exception as e:
        print(f'[fvd-viz] disabled (deps): {e}')
        return None

    cg, gg = load_clips_eval_grouped(gen_dir)
    ct, gt = load_clips_eval_grouped(tgt_dir)
    gi = {g: i for i, g in enumerate(gg)}
    ti = {g: i for i, g in enumerate(gt)}
    common = [g for g in gg if g in ti]          # gen/tgt 双方に存在する group のみ、gen 順
    if len(common) < 3:
        print(f'[fvd-viz] skip ({len(common)} common groups < 3)')
        return None
    try:
        fg = np.stack([extractor.feat(cg[gi[g]]) for g in common], axis=0)  # (N,D) gen
        ft = np.stack([extractor.feat(ct[ti[g]]) for g in common], axis=0)  # (N,D) tgt
    except Exception as e:
        print(f'[fvd-viz] feature extraction failed: {e}')
        return None
    N = len(common)

    # ---- 最近傍対応（検索）。index 一致 = 同じ group = 同一サンプル ----
    def retrieval(A, B):
        d = cdist(A, B)                          # (N,N) 行=query A, 列=DB B
        order = np.argsort(d, axis=1)
        nn = order[:, 0]
        rank = np.array([int(np.where(order[i] == i)[0][0]) for i in range(N)])
        return nn, rank, float(np.mean(nn == np.arange(N))), float(np.mean(rank < 5))
    nn_g2t, rank_g2t, t1_g2t, t5_g2t = retrieval(fg, ft)
    nn_t2g, rank_t2g, t1_t2g, t5_t2g = retrieval(ft, fg)
    print(f'[fvd-viz] {label} NN: gen->tgt top1={t1_g2t:.3f} top5={t5_g2t:.3f} | '
          f'tgt->gen top1={t1_t2g:.3f} top5={t5_t2g:.3f}  (N={N}, chance≈{1.0/N:.4f})')

    # ---- 2D 埋め込み (PCA, t-SNE) ----
    X = np.concatenate([fg, ft], axis=0)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    emb = {'pca': PCA(n_components=2).fit_transform(Xs)}
    try:
        from sklearn.manifold import TSNE
        perp = max(5, min(30, (2 * N - 1) // 3))
        emb['tsne'] = TSNE(n_components=2, perplexity=perp, init='pca',
                           random_state=0).fit_transform(Xs)
    except Exception as e:
        print(f'[fvd-viz] tsne skipped: {e}')

    os.makedirs(out_dir, exist_ok=True)
    for name, e2 in emb.items():
        eg, et = e2[:N], e2[N:]
        plt.figure(figsize=(6, 6))
        plt.scatter(eg[:, 0], eg[:, 1], s=12, c='tab:red', alpha=0.6, label='generated')
        plt.scatter(et[:, 0], et[:, 1], s=12, c='tab:blue', alpha=0.6, label='target')
        plt.legend(); plt.title(f'FVD feature space ({name.upper()}) - {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fvd_feat_{name}_{label}.png'), dpi=130)
        plt.close()

    # ---- 最近傍を線でつなぐ（緑=対応一致, 赤=不一致）----
    if 'tsne' in emb:
        e2 = emb['tsne']; eg, et = e2[:N], e2[N:]
        plt.figure(figsize=(7, 7))
        for i in range(N):
            j = int(nn_g2t[i])
            plt.plot([eg[i, 0], et[j, 0]], [eg[i, 1], et[j, 1]],
                     c=('tab:green' if j == i else 'tab:red'), lw=0.5, alpha=0.5)
        plt.scatter(eg[:, 0], eg[:, 1], s=10, c='black', label='gen')
        plt.scatter(et[:, 0], et[:, 1], s=10, c='tab:blue', label='tgt')
        plt.legend(); plt.title(f'gen->tgt NN (green=correct, red=wrong) - {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fvd_nn_lines_{label}.png'), dpi=130)
        plt.close()

    # ---- マッチ例モンタージュ（中央フレーム: gen | NN-tgt | true-tgt）----
    try:
        # 並べるサンプルは run 間で固定（生成ノイズで顔ぶれが変わらないよう nn_g2t に依存させない）。
        # 全 group から seed=0 でランダムに 12 個選ぶ。common 順は run 間で安定なので毎回同じ group。
        k = min(N, 12)
        pick = sorted(np.random.default_rng(0).choice(N, size=k, replace=False).tolist())
        mid = lambda clip: clip[len(clip) // 2]
        rows = [np.concatenate([mid(cg[gi[common[i]]]),
                                mid(ct[ti[common[int(nn_g2t[i])]]]),
                                mid(ct[ti[common[i]]])], axis=1) for i in pick]
        sheet = np.concatenate(rows, axis=0)
        plt.figure(figsize=(4, 1.1 * len(pick)))
        plt.imshow(sheet, cmap='gray'); plt.axis('off')
        plt.title('gen | NN-tgt | true-tgt')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fvd_nn_montage_{label}.png'), dpi=130)
        plt.close()
    except Exception as e:
        print(f'[fvd-viz] montage skipped: {e}')

    report = {
        'label': label, 'N': N, 'chance_top1': 1.0 / N,
        'gen2tgt_top1': t1_g2t, 'gen2tgt_top5': t5_g2t,
        'tgt2gen_top1': t1_t2g, 'tgt2gen_top5': t5_t2g,
        'gen2tgt_mean_rank': float(rank_g2t.mean()),
        'tgt2gen_mean_rank': float(rank_t2g.mean()),
        'groups': common,
        'gen2tgt_nn_group': [common[int(j)] for j in nn_g2t],
    }
    with open(os.path.join(out_dir, f'fvd_feature_report_{label}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[fvd-viz] saved figures + json to {out_dir}')
    return report


# --------------------------------------------------------------------------
# matching accuracy（sequence OT plan vs 対角 GT）
# --------------------------------------------------------------------------
def matching_accuracy(fake_seq, real_seq, ot_fn, reg, mono_pen, normalize, sinkhorn_type):
    """fake_seq:(T,C,H,W) 生成全フレーム, real_seq:(T_B,C,H,W) ターゲット(穴含む)。
       戻り値: dict or None"""
    try:
        _, details = ot_fn(
            fake_seq, real_seq,
            reg=reg,
            monotone=mono_pen > 0,
            monotone_penalty=mono_pen,
            normalize=normalize,
            sinkhorn_type=sinkhorn_type,
            return_details=True,
        )
    except Exception as e:
        print(f'[matching] skip (OT failed): {e}')
        return None

    P = details['P'].detach()                      # (T, N)
    vidx = details['valid_idx'].detach().cpu()     # (N,) 残った原フレーム index（昇順）
    T = P.shape[0]
    vidx_list = vidx.tolist()
    orig_of_col = {col: orig for col, orig in enumerate(vidx_list)}  # 列 -> 原 index
    valid_set = set(vidx_list)

    # source -> target: 各生成フレーム i（非穴のものだけ GT あり）の argmax 列が
    # 原 index i を指すか
    pred_col = P.argmax(dim=1).cpu()               # (T,)
    correct, total = 0, 0
    for i in range(T):
        if i in valid_set:
            total += 1
            if orig_of_col[int(pred_col[i])] == i:
                correct += 1
    acc_s2t = correct / total if total > 0 else float('nan')

    # target -> source: 各ターゲット列 k の argmax 行が原 index vidx[k] か
    pred_row = P.argmax(dim=0).cpu()               # (N,)
    c2, t2 = 0, 0
    for k, orig in enumerate(vidx_list):
        t2 += 1
        if int(pred_row[k]) == orig:
            c2 += 1
    acc_t2s = c2 / t2 if t2 > 0 else float('nan')

    return {
        'match_acc_src2tgt': acc_s2t,
        'match_acc_tgt2src': acc_t2s,
        'num_valid_frames': total,
        'num_total_frames': T,
    }


def main():
    argv = sys.argv
    kid_subset_cap = int(_pop_arg(argv, '--eval_kid_subset', 900))
    eval_out = _pop_arg(argv, '--eval_out', '')
    lpips_net = _pop_arg(argv, '--eval_lpips_net', 'alex')
    match_file_B = _pop_arg(argv, '--eval_match_file_B', '')
    ref_fid = _pop_arg(argv, '--eval_ref_fid', '1') not in ('0', 'false', 'False')
    do_fvd = _pop_arg(argv, '--eval_fvd', '1') not in ('0', 'false', 'False')
    fvd_backend = _pop_arg(argv, '--eval_fvd_backend', 'i3d').lower()
    fvd_i3d_path = _pop_arg(
        argv, '--eval_fvd_i3d',
        os.environ.get('FVD_I3D_PATH', '/workspace/.cache/fvd/i3d_torchscript.pt'))
    do_fvd_viz = _pop_arg(argv, '--eval_fvd_viz', '0') not in ('0', 'false', 'False')
    do_paired = _pop_arg(argv, '--eval_paired', '1') not in ('0', 'false', 'False')
    txt_name = _pop_arg(argv, '--eval_txt_name', 'evaluation_metrics.txt')
    if not txt_name.endswith('.txt'):
        txt_name += '.txt'
    n_samples = int(_pop_arg(argv, '--eval_save_samples', 16))

    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if (len(opt.gpu_ids) > 0 and torch.cuda.is_available()) else 'cpu')

    # OT（matching accuracy 用）。無い codebase（UNSB-main）では None
    try:
        from models.sequence_ot import sequence_ot_loss_torch as ot_fn
    except Exception:
        ot_fn = None
    is_seq_model = (getattr(opt, 'model', '') == 'wpsb') and (ot_fn is not None)

    # LPIPS
    try:
        import lpips as lpips_lib
        lpips_model = lpips_lib.LPIPS(net=lpips_net).to(device).eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
    except Exception as e:
        print(f'[lpips] disabled: {e}')
        lpips_model = None

    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)

    model = create_model(opt)

    # matching accuracy 専用の穴あきターゲット（任意）。
    # MovingMnistPairedDataset と同じ transpose / phase split を再現して、
    # ループのローカル index と揃える（train_end=1000, val_end=1200 固定）。
    match_arr = None
    if is_seq_model and match_file_B:
        path = match_file_B if os.path.isabs(match_file_B) else os.path.join(
            getattr(opt, 'dataroot_B', '') or opt.dataroot, match_file_B)
        arr = np.load(path)
        if arr.shape[0] == 20:
            arr = np.transpose(arr, (1, 0, 2, 3))  # (N,T,H,W)
        if opt.phase == 'train':
            match_arr = arr[:1000]
        elif opt.phase == 'val':
            match_arr = arr[1000:1200]
        else:
            match_arr = arr[1200:]
        print(f'[matching] using holed target: {path}  (split {opt.phase}, {match_arr.shape[0]} seq)')

    def build_seq_tensor(seq_np, device):
        """(T,H,W) uint8/float -> (T,1,H,W) in [-1,1]（dataset の grayscale transform 同等）"""
        x = torch.from_numpy(np.asarray(seq_np)).float()
        if x.max() > 1:
            x = x / 255.0
        x = x * 2 - 1                      # [0,1] -> [-1,1]
        return x.unsqueeze(1).to(device)   # (T,1,H,W)

    # FID/KID 用の一時 PNG ディレクトリ
    tmp_root = tempfile.mkdtemp(prefix='evalfid_')
    fake_dir = os.path.join(tmp_root, 'fake')
    real_dir = os.path.join(tmp_root, 'real')
    src_dir = os.path.join(tmp_root, 'src')

    # ペア指標アキュムレータ（フレーム重み付き平均）
    sums = {'SSIM': 0.0, 'PSNR': 0.0, 'L1': 0.0, 'L2': 0.0, 'LPIPS': 0.0}
    n_frames = 0
    match_sums = {'src2tgt': 0.0, 'tgt2src': 0.0}
    n_match = 0
    fake_count = real_count = src_count = 0
    fake_source = 'fake_B'
    seq_tags = set()
    per_seq = []

    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(dataset, dataset2)):
            if i == 0:
                try:
                    model.data_dependent_initialize(data, data2)
                except TypeError:
                    model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:
                break

            try:
                model.set_input(data, data2)
            except TypeError:
                model.set_input(data)
            model.test()

            if not (hasattr(model, 'fake_B') and hasattr(model, 'real_B')):
                continue

            # test forward は fake_1..fake_T（フル chain の NFE 別出力）を持つ。
            # fake_B は学習パスのランダム深さ出力なので、最終 NFE 出力を優先する。
            n_steps = int(getattr(opt, 'num_timesteps', 0))
            fake_final = getattr(model, f'fake_{n_steps}', None)
            fake_source = f'fake_{n_steps}' if fake_final is not None else 'fake_B'
            fake = flatten_seq(fake_final if fake_final is not None else model.fake_B).to(device)   # (T,C,H,W)
            real = flatten_seq(model.real_B).to(device)   # (T_B,C,H,W)

            vidx = valid_frame_idx(real)
            fake_v = fake[vidx]
            real_v = real[vidx]

            # ペア指標（--eval_paired 0 で丸ごとスキップ。分布/matching は影響なし）
            bs = fake_v.size(0)
            s = ps = l1 = l2 = lp = float('nan')
            if do_paired:
                s = _ssim(fake_v, real_v)
                ps = _psnr(fake_v, real_v)
                l1 = _l1(fake_v, real_v)
                l2 = _l2(fake_v, real_v)
                sums['SSIM'] += s * bs
                sums['PSNR'] += ps * bs
                sums['L1'] += l1 * bs
                sums['L2'] += l2 * bs
                if lpips_model is not None:
                    f3 = fake_v if fake_v.size(1) == 3 else fake_v.repeat(1, 3, 1, 1)
                    r3 = real_v if real_v.size(1) == 3 else real_v.repeat(1, 3, 1, 1)
                    lp = lpips_model(f3.clamp(-1, 1), r3.clamp(-1, 1)).mean().item()
                    sums['LPIPS'] += lp * bs
            n_frames += bs

            # 分布指標用に書き出し（group = 系列 ID。フレーム単位 dataset でも系列でまとまる）
            tag = seq_tag_of(data, f'i{i:06d}')
            seq_tags.add(tag)
            fake_count = save_frames(fake_v, fake_dir, tag, fake_count)
            real_count = save_frames(real_v, real_dir, tag, real_count)
            if ref_fid and hasattr(model, 'real_A'):
                src = flatten_seq(model.real_A).to(device)
                src_count = save_frames(src, src_dir, tag, src_count)

            # matching accuracy
            seq_rec = {'index': i, 'SSIM': s, 'PSNR': ps, 'L1': l1, 'L2': l2, 'LPIPS': lp}
            if is_seq_model:
                match_real = real
                if match_arr is not None and i < match_arr.shape[0]:
                    match_real = build_seq_tensor(match_arr[i], device)
                ma = matching_accuracy(
                    fake, match_real, ot_fn,
                    reg=getattr(opt, 'lmda', 0.05),
                    mono_pen=getattr(opt, 'seq_ot_monotone_penalty', 0.0),
                    normalize=getattr(opt, 'seq_ot_normalize', 'mean'),
                    sinkhorn_type=getattr(opt, 'sinkhorn_type', 'sinkhorn'),
                )
                if ma is not None:
                    match_sums['src2tgt'] += ma['match_acc_src2tgt']
                    match_sums['tgt2src'] += ma['match_acc_tgt2src']
                    n_match += 1
                    seq_rec.update(ma)
            per_seq.append(seq_rec)

            if i % 10 == 0:
                print(f'[{i}] SSIM={s:.4f} PSNR={ps:.2f} L1={l1:.4f} LPIPS={lp:.4f}')

    if n_frames == 0:
        print('No frames evaluated. (fake_B/real_B not found?)')
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    if do_paired:
        paired = {k: v / n_frames for k, v in sums.items()}
        if lpips_model is None:
            paired['LPIPS'] = None
    else:
        paired = None

    # ---- FID / KID（torch_fidelity、複数ペア）----
    dist = {}

    def fid_kid(d1, d2, label):
        try:
            import torch_fidelity
            n1 = len(os.listdir(d1))
            n2 = len(os.listdir(d2))
            kid_subset = max(2, min(kid_subset_cap, n1, n2))
            m = torch_fidelity.calculate_metrics(
                input1=d1, input2=d2,
                cuda=(device.type == 'cuda'),
                fid=True, kid=True,
                kid_subset_size=kid_subset,
                verbose=False,
            )
            print(f'[fid/kid] {label}: FID={m["frechet_inception_distance"]:.4f}')
            return {
                'FID': float(m['frechet_inception_distance']),
                'KID_mean': float(m['kernel_inception_distance_mean']),
                'KID_std': float(m['kernel_inception_distance_std']),
                'n1': n1, 'n2': n2,
            }
        except Exception as e:
            print(f'[fid/kid] {label} failed: {e}')
            return None

    # FVD 抽出器（r3d_18）。FID と同じディレクトリ対を系列単位で束ね直して使う。
    fvd_extractor = None
    fvd_backend_used = None
    if do_fvd:
        if fvd_backend == 'i3d':
            try:
                fvd_extractor = I3DExtractor(device, fvd_i3d_path)
                fvd_backend_used = 'i3d'
                print(f'[fvd] extractor: I3D (Kinetics-400, official-compatible), '
                      f'paper-comparable; weights={fvd_i3d_path}')
            except Exception as e:
                print(f'[fvd] I3D unavailable ({e}); falling back to r3d_18 (relative-only)')
        if fvd_extractor is None:
            try:
                fvd_extractor = FVDExtractor(device)
                fvd_backend_used = 'r3d18'
                print('[fvd] extractor: r3d_18 (Kinetics-400), 512-dim, frozen '
                      '(relative-only, NOT paper-comparable)')
            except Exception as e:
                print(f'[fvd] disabled: {e}')

    def dist_pair(key, d1, d2, label):
        """同じディレクトリ対で FID/KID と FVD を計算し、1 つの dict にまとめる。"""
        entry = fid_kid(d1, d2, label)
        fv = compute_fvd(fvd_extractor, d1, d2, label)
        if fv is not None:
            entry = dict(entry or {})
            entry['FVD'] = fv['FVD']
            # FVD は系列(クリップ)単位なので FID のフレーム数 n1/n2 とは別に持つ
            entry['FVD_n1'] = fv['n1']
            entry['FVD_n2'] = fv['n2']
        dist[key] = entry

    dist_pair('gen_vs_tgt', fake_dir, real_dir, 'gen vs tgt')
    if ref_fid and src_count > 0:
        dist_pair('gen_vs_src', fake_dir, src_dir, 'gen vs src')
        dist_pair('src_vs_tgt', src_dir, real_dir, 'src vs tgt')
        s1, s2 = half_split_by_seq(src_dir, tmp_root, 'src')
        dist_pair('src_vs_src', s1, s2, 'src vs src (half)')
        t1, t2 = half_split_by_seq(real_dir, tmp_root, 'tgt')
        dist_pair('tgt_vs_tgt', t1, t2, 'tgt vs tgt (half)')

    main_d = dist.get('gen_vs_tgt') or {}
    fid = main_d.get('FID')
    kid_mean = main_d.get('KID_mean')
    kid_std = main_d.get('KID_std')

    matching = None
    if n_match > 0:
        matching = {
            'match_acc_src2tgt': match_sums['src2tgt'] / n_match,
            'match_acc_tgt2src': match_sums['tgt2src'] / n_match,
            'num_sequences': n_match,
            'match_target': match_file_B if match_file_B else getattr(opt, 'data_file_B', ''),
        }

    result = {
        'name': opt.name,
        'epoch': opt.epoch,
        'phase': opt.phase,
        'num_sequences': len(seq_tags),
        'num_items': len(per_seq),
        'num_frames': n_frames,
        'fake_source': fake_source,
        'data_file_B': getattr(opt, 'data_file_B', ''),
        'paired': paired,
        'distribution': {'pairs': dist, 'fvd_backend': fvd_backend_used,
                         'num_fake': fake_count, 'num_real': real_count, 'num_src': src_count},
        'matching': matching,
        'per_sequence': per_seq,
    }

    # ---- 保存先 ----
    if not eval_out:
        eval_out = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(eval_out, exist_ok=True)

    # ---- FVD 特徴空間の可視化 + 最近傍対応（--eval_fvd_viz 1）----
    # tmp の fake_dir/real_dir はまだ生きている（最後の rmtree 前）ので同じ PNG を再利用する。
    if do_fvd_viz and fvd_extractor is not None and fake_count > 0 and real_count > 0:
        rep = fvd_feature_report(fvd_extractor, fake_dir, real_dir,
                                 os.path.join(eval_out, 'fvd_viz'), 'gen_vs_tgt')
        if rep is not None:
            result['distribution']['fvd_nn'] = {
                k: rep[k] for k in ('N', 'chance_top1', 'gen2tgt_top1', 'gen2tgt_top5',
                                    'tgt2gen_top1', 'tgt2gen_top5',
                                    'gen2tgt_mean_rank', 'tgt2gen_mean_rank')}

    json_path = os.path.join(eval_out, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # ---- 可読テキスト ----
    lines = []
    lines.append('=' * 60)
    lines.append(f'Evaluation: {opt.name}  (epoch {opt.epoch}, phase {opt.phase})')
    lines.append(f'sequences={len(seq_tags)}  frames={n_frames}  GT_B={result["data_file_B"]}  output={fake_source}')
    lines.append('=' * 60)
    if paired is not None:
        lines.append('[Paired metrics] (non-holed diagonal GT)')
        lp_str = f'{paired["LPIPS"]:.4f}' if paired['LPIPS'] is not None else 'n/a'
        lines.append(f'  SSIM  : {paired["SSIM"]:.4f}   (higher better)')
        lines.append(f'  PSNR  : {paired["PSNR"]:.4f} dB(higher better)')
        lines.append(f'  L1    : {paired["L1"]:.6f}   (lower better)')
        lines.append(f'  L2    : {paired["L2"]:.6f}   (lower better)')
        lines.append(f'  LPIPS : {lp_str}   (lower better)')
    else:
        lines.append('[Paired metrics] skipped (--eval_paired 0)')
    lines.append('[Distribution metrics] (torch_fidelity, same test sequences)')
    pair_desc = [
        ('gen_vs_tgt', 'gen vs tgt   (main, lower better)'),
        ('tgt_vs_tgt', 'tgt vs tgt   (floor: same-dist halves)'),
        ('src_vs_src', 'src vs src   (floor: same-dist halves)'),
        ('src_vs_tgt', 'src vs tgt   (upper ref: identity map)'),
        ('gen_vs_src', 'gen vs src   (distance from source)'),
    ]
    for key, desc in pair_desc:
        d = dist.get(key)
        if d is None:
            continue
        lines.append(f'  {desc}')
        if 'FID' in d:
            lines.append(f'      FID={d["FID"]:.4f}  KID={d["KID_mean"]:.6f}+/-{d["KID_std"]:.6f}  (n={d["n1"]}/{d["n2"]})')
        if 'FVD' in d:
            fvd_tag = ('I3D, paper-comparable' if fvd_backend_used == 'i3d'
                       else 'r3d_18, relative-only')
            lines.append(f'      FVD={d["FVD"]:.4f}  (clip-level, {fvd_tag}; n={d.get("FVD_n1")}/{d.get("FVD_n2")})')
    if not dist:
        lines.append('  FID/KID: n/a')
    nn = result['distribution'].get('fvd_nn')
    if nn is not None:
        lines.append('[FVD feature NN retrieval] (gen feat -> nearest tgt feat = same sample?)')
        lines.append(f'  gen->tgt : top1={nn["gen2tgt_top1"]:.3f}  top5={nn["gen2tgt_top5"]:.3f}  '
                     f'mean_rank={nn["gen2tgt_mean_rank"]:.2f}')
        lines.append(f'  tgt->gen : top1={nn["tgt2gen_top1"]:.3f}  top5={nn["tgt2gen_top5"]:.3f}  '
                     f'mean_rank={nn["tgt2gen_mean_rank"]:.2f}')
        lines.append(f'  (N={nn["N"]}, chance top1={nn["chance_top1"]:.4f}; higher better)')
    if matching is not None:
        lines.append('[Matching accuracy] (sequence OT plan vs diagonal GT)')
        lines.append(f'  src->tgt : {matching["match_acc_src2tgt"]:.4f}')
        lines.append(f'  tgt->src : {matching["match_acc_tgt2src"]:.4f}')
    lines.append('=' * 60)
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(os.path.join(eval_out, txt_name), 'w') as f:
        f.write(txt + '\n')

    # ---- サンプルシート（src | gen | tgt を並べた確認用画像）----
    # FID に渡したものと同一の PNG から作るので、ディレクトリ取り違えの検査になる。
    if n_samples > 0:
        try:
            fake_files = sorted(os.listdir(fake_dir))
            real_files = sorted(os.listdir(real_dir))
            src_files = sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []
            step = max(1, len(fake_files) // n_samples)
            rows = []
            for k in list(range(0, len(fake_files), step))[:n_samples]:
                row = []
                if src_files:
                    row.append(np.array(Image.open(os.path.join(src_dir, src_files[k]))))
                row.append(np.array(Image.open(os.path.join(fake_dir, fake_files[k]))))
                row.append(np.array(Image.open(os.path.join(real_dir, real_files[k]))))
                rows.append(np.concatenate(row, axis=1))
            sheet = np.concatenate(rows, axis=0)
            cols = ('src|gen|tgt' if src_files else 'gen|tgt')
            sheet_path = os.path.join(eval_out, f'samples_{cols.replace("|", "_")}.png')
            Image.fromarray(sheet).save(sheet_path)
            print(f'Sample sheet ({cols}): {sheet_path}')
        except Exception as e:
            print(f'[samples] failed: {e}')

    print(f'\nSaved: {json_path}')
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == '__main__':
    main()
