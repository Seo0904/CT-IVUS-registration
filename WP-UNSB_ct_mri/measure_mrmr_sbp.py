"""
MR-MR SB_P floor measurement (flatten / D-feature space).

学習(flatten実験 20260705_155621)と同じ経路で SB_P(=ot_cost) を測る。
  - netD(basic_cond) を epoch チェックポイントからロード
  - model.seq_ot_descriptors で D 特徴 (T,D) を抽出
  - sequence_ot_loss(solver=geo, reg=0.05, iters=30, debias=True, ...) の ot_cost を SB_P とする

2 ケース:
  (a) same : 同一患者 MR を fake/tgt 両方に入れる (=完璧生成の理論フロア。valid_idx 非対称ぶんは残る)
  (b) cross: 別患者 MR を fake 側に入れる (モダリティギャップ0で残る解剖差フロア)

time_idx (0..num_timesteps-1) はランダム1時刻で D 条件付け → 全時刻で測って平均/レンジも出す。
"""
import os
import argparse
import types
import numpy as np
import torch

from models import create_model
from models.sequence_ot import sequence_ot_loss

EXP_DIR = "/workspace/data/experiment_result/WP-UNSB_ct_mri/synthRAD-brain/20260705_155621"
EXP_NAME = "synthRAD_brain_ct2mr_sb_otdiv_geo_dfeat_flatten_gan"
DATAROOT = "/workspace/data/preprocessed/synthRAD_maskfree_full"


def _cast(v):
    v = v.strip()
    if v == "True":
        return True
    if v == "False":
        return False
    if v == "None" or v == "":
        return None if v == "None" else ""
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_opt(train_opt_path):
    opt = types.SimpleNamespace()
    with open(train_opt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            # 値は最初の [default...] より前まで
            val = rest.split("[default")[0].strip()
            setattr(opt, key, _cast(val))
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="400")
    ap.add_argument("--phase", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n_patients", type=int, default=0, help="0=all in phase")
    ap.add_argument("--n_cross", type=int, default=3, help="別患者ペア数/患者")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    opt = load_opt(os.path.join(EXP_DIR, EXP_NAME, "train_opt.txt"))
    # 実行環境に合わせて上書き
    opt.gpu_ids = [0]
    opt.isTrain = True          # netD を生成させるため
    opt.phase = args.phase
    opt.checkpoints_dir = EXP_DIR
    opt.name = EXP_NAME
    opt.dataroot = DATAROOT
    opt.h5_path = ""
    opt.split_path = ""
    opt.pretrained_name = None
    opt.continue_train = False
    opt.verbose = False
    opt.suffix = ""

    device = torch.device("cuda:0")

    # ---- モデル生成 & D ロード ----
    model = create_model(opt)
    model.setup_done = True
    # networks は create_model 内では作られるが load はしない実装があるので明示 load
    model.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model.load_networks(args.epoch)
    model.netD.eval()
    for p in model.netD.parameters():
        p.requires_grad_(False)

    T_steps = int(opt.num_timesteps)
    reg = float(opt.lmda)
    iters = int(opt.seq_ot_iters)

    # ---- MR volume 読み込み (dataset と同じ正規化) ----
    from data.ct_mri_dataset import CtMriDataset
    ds = CtMriDataset(opt)
    n_pat = len(ds)
    idxs = list(range(n_pat))
    if args.n_patients > 0:
        idxs = idxs[: args.n_patients]

    print(f"[cfg] epoch={args.epoch} phase={args.phase} patients={len(idxs)}/{n_pat} "
          f"reg={reg} iters={iters} T_steps={T_steps} debias={opt.seq_ot_debias} "
          f"cost_space={opt.seq_ot_cost_space}")

    # 各患者の MR sequence をキャッシュ (S,1,H,W)
    mr = {}
    for i in idxs:
        mr[i] = ds[i]["B"].to(device)  # (S,1,H,W) in [-1,1]

    def sbp(fake_img, tgt_img):
        """fake/tgt image seq (S,1,H,W) -> SB_P(ot_cost), 全時刻平均"""
        vals = []
        for t in range(T_steps):
            model.time_idx = torch.full((1,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                ff, tf = model.seq_ot_descriptors(fake_img, tgt_img)
                _, terms = sequence_ot_loss(
                    fake_img, tgt_img,
                    fake_feat=ff, tgt_feat=tf,
                    solver="geo", reg=reg, iters=iters,
                    monotone=False, P_entropy=False, ot_divergence=False,
                    normalize=None, geo_p=int(opt.seq_ot_geo_p),
                    unbalanced=None,
                    debias=bool(opt.seq_ot_debias),
                )
            vals.append(float(terms["ot_cost"]))
        return vals  # len T_steps

    same_all = []   # 患者ごと (全時刻平均)
    cross_all = []
    print("\n patient        same(t-avg)   cross(t-avg)")
    for i in idxs:
        s_vals = sbp(mr[i], mr[i])
        same_mean = float(np.mean(s_vals))
        same_all.append(same_mean)

        # cross: 別患者を fake 側に
        others = [j for j in idxs if j != i]
        pick = np.random.choice(others, size=min(args.n_cross, len(others)), replace=False)
        c_means = []
        for j in pick:
            c_vals = sbp(mr[j], mr[i])  # fake=別患者MR_j, tgt=MR_i
            c_means.append(float(np.mean(c_vals)))
        cross_mean = float(np.mean(c_means))
        cross_all.append(cross_mean)
        cid = ds.cases[i]
        print(f" {cid:>8}   {same_mean:12.6f}  {cross_mean:12.6f}")

    print("\n================ SUMMARY (SB_P = ot_cost, flatten/D-feat) ================")
    print(f"epoch={args.epoch}  reg={reg} iters={iters}  debias={opt.seq_ot_debias}")
    print(f"  same-patient  MR-MR : mean={np.mean(same_all):.6f}  std={np.std(same_all):.6f}  "
          f"min={np.min(same_all):.6f}  max={np.max(same_all):.6f}   (n={len(same_all)})")
    print(f"  cross-patient MR-MR : mean={np.mean(cross_all):.6f}  std={np.std(cross_all):.6f}  "
          f"min={np.min(cross_all):.6f}  max={np.max(cross_all):.6f}   (n={len(cross_all)})")
    print("==========================================================================")


if __name__ == "__main__":
    main()
