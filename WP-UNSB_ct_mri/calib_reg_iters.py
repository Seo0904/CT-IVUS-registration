"""
D 特徴空間 seq-OT の reg(lmda)/iters を実データで較正するスクリプト。

前提: seq_ot_cost_space=flatten, seq_ot_feat_norm=l2 (単位球面化)。
このとき M のスケールは metric ごとに固定される:
    cosine : M = 1 - x̂·ŷ            ∈ [0, 2]
    l2     : M = ‖x̂ - ŷ‖²           ∈ [0, 4]
D_loss_mode(ot/gan) はコスト行列のスケールに影響しないので、reg/iters の
較正軸は metric(l2/cosine) の 2 つで足りる(G側SBもot時のD側も同じ reg/iters を使う)。

各 (metric, reg, iters) について測る指標:
  P_IDENT_ERR : 行正規化した輸送計画 P と単位行列 I の Frobenius 誤差。
                小さいほど「1対1(対角)対応」= 望ましい。
  GRAD_NORM   : ot_cost の fake_seq に対する勾配ノルム。0 だと勾配消失(学習不能)。
  OT_COST     : Sinkhorn divergence(debias)の値。
  CONV(iters) : iters を増やして ot_cost が変化しなくなれば収束。

較正の目標(ユーザー基準):
  P_IDENT_ERR が小さく(1対1)、GRAD_NORM が有限で消失せず、
  かつ reg が小さく iters も小さいスイートスポットを metric ごとに選ぶ。
"""
import sys
import numpy as np
import torch

DATAROOT = "/workspace/data/preprocessed/synthRAD_maskfree_full"

# --- opt を組み立てる(学習と同じ経路で parse) ---
sys.argv = [
    "calib",
    "--dataroot", DATAROOT,
    "--region", "brain",
    "--model", "wpsb",
    "--dataset_mode", "ct_mri",
    "--name", "calib_reg_iters",
    "--input_nc", "1", "--output_nc", "1",
    "--ngf", "64", "--ndf", "64",
    "--num_timesteps", "5",
    "--load_size", "128", "--crop_size", "128",
    "--preprocess", "none", "--no_flip",
    "--batch_size", "1",
    "--gpu_ids", "0",
    "--num_slices_per_seq", "16",     # 較正は軽く間引く(スケール感は不変)
    "--num_threads", "0",
    "--sb_mode", "seq_ot",
    "--seq_ot_solver", "geo",
    "--seq_ot_normalize", "none",
    "--seq_ot_cost_space", "flatten",
    "--seq_ot_feat_norm", "l2",
    "--lambda_NCE", "0",
    "--lambda_GAN", "0",
    "--checkpoints_dir", "/tmp/claude-1000/-workspace/03bb2a6d-4f19-4fd7-a0de-97088b87abc3/scratchpad/calib_ckpt",
    "--display_id", "0",
]

from options.train_options import TrainOptions
opt = TrainOptions().parse()

from data import create_dataset
from models import create_model
from models.sequence_ot import sequence_ot_loss

torch.manual_seed(0)
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.netG.eval()
model.netD.eval()   # BN を固定してスケールを安定させる(較正目的)
device = model.device

# --- 実データ数シーケンスから「対応構造を持つ合成 fake」を作る ---
# 学習前の netG 出力は real と無関係でコスト行列に対角構造が無く、
# reg を下げても P が I に近づかない(=1対1 を評価できない)。
# そこで fake = real_B + σ·noise とし、対応フレーム同士が最小コストになる
# 対角構造を人工的に作る。これで「reg を下げると 1対1(P_IDENT_ERR↓)が
# 復元されるか」「その reg で勾配が流れるか」を測れる。学習が進むと実 fake は
# この状態に近づくので、較正として妥当。
SIGMA = 0.1
N_SEQ = 3
descriptors = []   # (ff[grad], tf[nograd], fake_syn[grad])

it = iter(dataset)
for s in range(N_SEQ):
    data = next(it)
    model.set_input(data)
    real_seq = model.real_B[0] if model.real_B.dim() == 5 else model.real_B  # (T,C,H,W)
    model.time_idx = torch.zeros(1, device=device, dtype=torch.long)

    fake_syn = (real_seq.detach() + SIGMA * torch.randn_like(real_seq)).clamp(-1, 1)
    fake_syn = fake_syn.requires_grad_(True)

    fake_flat, meta = model._flatten_seq_for_net(fake_syn)
    _, feat_fake = model.netD(fake_flat, model.time_idx, return_feat=True)
    feat_fake_seq = model._restore_feat_seq(feat_fake, meta)
    ff = model._seq_descriptor(feat_fake_seq, None)   # (T, D) grad保持
    with torch.no_grad():
        real_flat, rmeta = model._flatten_seq_for_net(real_seq)
        _, feat_real = model.netD(real_flat, model.time_idx, return_feat=True)
        feat_real_seq = model._restore_feat_seq(feat_real, rmeta)
        tf = model._seq_descriptor(feat_real_seq, None)
    descriptors.append((ff, tf, fake_syn))
    print(f"[seq {s}] T={ff.shape[0]} D={ff.shape[1]}  fake_syn(real+{SIGMA}noise)={tuple(fake_syn.shape)}")

REGS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ITERS = [20, 50, 100, 200]

def eval_one(metric, reg, iters, debias):
    """N_SEQ 平均の P_IDENT_ERR / GRAD_NORM / OT_COST を返す。"""
    pid, gnorm, cost = [], [], []
    for ff, tf, fake_syn in descriptors:
        ot_val, terms = sequence_ot_loss(
            fake_syn, fake_syn,           # tgt_seq は valid_idx 用ダミー(全frame有効)
            fake_feat=ff, tgt_feat=tf,
            solver="geo", reg=reg, iters=iters,
            monotone=False, P_entropy=False, ot_divergence=False,
            normalize=None, debias=debias,
            metric=metric, feat_norm="l2",
            return_details=False,
        )
        g = torch.autograd.grad(ot_val, fake_syn, retain_graph=True, allow_unused=True)[0]
        gn = float(g.norm()) if g is not None else 0.0
        pid.append(float(terms["P_ident_err"]))
        gnorm.append(gn)
        cost.append(float(ot_val))
    return np.mean(pid), np.mean(gnorm), np.mean(cost)

# 系統: (metric, debias)。debias=True は l2/cosine 等価なので cosine 1本。
SETUPS = [
    ("cosine", True,  "cosine + debias=True (=l2 と等価/Sinkhorn divergence)"),
    ("cosine", False, "cosine + debias=False (biased OT, OT-GAN準拠)"),
    ("l2",     False, "l2(cdist²) + debias=False (biased OT)"),
]

for metric, debias, title in SETUPS:
    print(f"\n========== {title} | feat_norm=l2 ==========")
    print(f"{'reg':>7} | " + " ".join(f"it={it:>4}" for it in ITERS))
    print("  --- P_IDENT_ERR (小=1対1復元) ---")
    for reg in REGS:
        row = [f"{eval_one(metric, reg, it, debias)[0]:7.3f}" for it in ITERS]
        print(f"{reg:>7} | " + " ".join(row))
    print("  --- GRAD_NORM (0=消失) ---")
    for reg in REGS:
        row = [f"{eval_one(metric, reg, it, debias)[1]:.2e}" for it in ITERS]
        print(f"{reg:>7} | " + " ".join(f"{v:>8}" for v in row))
    print("  --- OT_COST (収束: iters増で不変か) ---")
    for reg in REGS:
        row = [f"{eval_one(metric, reg, it, debias)[2]:8.4f}" for it in ITERS]
        print(f"{reg:>7} | " + " ".join(row))

print("\nDONE")
