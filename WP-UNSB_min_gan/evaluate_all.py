"""
統合定量評価スクリプト（WP-UNSB / UNSB 共通）

best checkpoint をロードして test セットを再推論し、以下をまとめて算出する:

  1. ペア指標（フレーム単位 GT、非穴フレームの対角で評価）
       L1 / L2 / PSNR / SSIM / LPIPS
  2. 分布指標（torch_fidelity、同一の test 系列で統一）
       FID / KID(mean ± std) を以下のペアで算出:
         gen vs tgt（メイン指標）
         gen vs src（ソースから離れたか = 入力コピーの検出）
         src vs tgt（ドメイン間距離 = 恒等写像の上限リファレンス）
         src vs src / tgt vs tgt（系列単位の半分割 2 群 = FID の床）
  3. matching accuracy（WP-UNSB 系のみ）
       学習済み生成器に対し sequence OT で推定したフレーム間対応が
       「生成フレーム i ↔ ターゲット原フレーム i」の対角 GT とどれだけ一致するか

オプションは test.py / run_test_*.sh とまったく同じものを渡せる
（TestOptions をそのまま流用）。追加オプション:
  --eval_kid_subset N   KID のサブセットサイズ上限（default 900）
  --eval_out DIR        結果 JSON/txt の保存先（default: results_dir/name/phase_epoch）
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
import sys
import json
import tempfile
import shutil
from math import log10

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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

            # ペア指標
            bs = fake_v.size(0)
            s = _ssim(fake_v, real_v)
            ps = _psnr(fake_v, real_v)
            l1 = _l1(fake_v, real_v)
            l2 = _l2(fake_v, real_v)
            sums['SSIM'] += s * bs
            sums['PSNR'] += ps * bs
            sums['L1'] += l1 * bs
            sums['L2'] += l2 * bs
            lp = float('nan')
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

    paired = {k: v / n_frames for k, v in sums.items()}
    if lpips_model is None:
        paired['LPIPS'] = None

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

    dist['gen_vs_tgt'] = fid_kid(fake_dir, real_dir, 'gen vs tgt')
    if ref_fid and src_count > 0:
        dist['gen_vs_src'] = fid_kid(fake_dir, src_dir, 'gen vs src')
        dist['src_vs_tgt'] = fid_kid(src_dir, real_dir, 'src vs tgt')
        s1, s2 = half_split_by_seq(src_dir, tmp_root, 'src')
        dist['src_vs_src'] = fid_kid(s1, s2, 'src vs src (half)')
        t1, t2 = half_split_by_seq(real_dir, tmp_root, 'tgt')
        dist['tgt_vs_tgt'] = fid_kid(t1, t2, 'tgt vs tgt (half)')

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
        'distribution': {'pairs': dist,
                         'num_fake': fake_count, 'num_real': real_count, 'num_src': src_count},
        'matching': matching,
        'per_sequence': per_seq,
    }

    # ---- 保存先 ----
    if not eval_out:
        eval_out = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(eval_out, exist_ok=True)
    json_path = os.path.join(eval_out, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # ---- 可読テキスト ----
    lines = []
    lines.append('=' * 60)
    lines.append(f'Evaluation: {opt.name}  (epoch {opt.epoch}, phase {opt.phase})')
    lines.append(f'sequences={len(seq_tags)}  frames={n_frames}  GT_B={result["data_file_B"]}  output={fake_source}')
    lines.append('=' * 60)
    lines.append('[Paired metrics] (non-holed diagonal GT)')
    lp_str = f'{paired["LPIPS"]:.4f}' if paired['LPIPS'] is not None else 'n/a'
    lines.append(f'  SSIM  : {paired["SSIM"]:.4f}   (higher better)')
    lines.append(f'  PSNR  : {paired["PSNR"]:.4f} dB(higher better)')
    lines.append(f'  L1    : {paired["L1"]:.6f}   (lower better)')
    lines.append(f'  L2    : {paired["L2"]:.6f}   (lower better)')
    lines.append(f'  LPIPS : {lp_str}   (lower better)')
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
        lines.append(f'      FID={d["FID"]:.4f}  KID={d["KID_mean"]:.6f}+/-{d["KID_std"]:.6f}  (n={d["n1"]}/{d["n2"]})')
    if not dist:
        lines.append('  FID/KID: n/a')
    if matching is not None:
        lines.append('[Matching accuracy] (sequence OT plan vs diagonal GT)')
        lines.append(f'  src->tgt : {matching["match_acc_src2tgt"]:.4f}')
        lines.append(f'  tgt->src : {matching["match_acc_tgt2src"]:.4f}')
    lines.append('=' * 60)
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open(os.path.join(eval_out, 'evaluation_metrics.txt'), 'w') as f:
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
