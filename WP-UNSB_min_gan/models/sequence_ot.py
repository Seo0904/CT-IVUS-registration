# UNSB-main/models/sequence_ot.py
from sympy import false
import torch
import ot  # POT
from typing import cast, Optional, Dict, Any


def get_valid_frame_idx(seq: torch.Tensor) -> torch.Tensor:
    """
    seq: (T, C, H, W) in [-1, 1]
    ゼロ画素フレーム（正規化後 max == -1.0）を除いた有効フレームのインデックスを返す。
    全フレームがゼロの場合は全インデックスを返す。
    """
    frame_max = seq.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]  # (T,)
    valid = (frame_max > -1.0 + 1e-3).nonzero(as_tuple=False).squeeze(1)
    if valid.numel() == 0:
        valid = torch.arange(seq.shape[0], device=seq.device)
    return valid


def sequence_ot_loss_torch(
    fake_seq: torch.Tensor,
    tgt_seq: torch.Tensor,
    reg: float = 0.05,
    iters: int = 50,
    monotone: bool = True,
    monotone_penalty: float = 50.0,
    P_entropy: bool = False,
    P_entropy_penalty: float = 1.0,
    ot_divergence: bool = False,
    ot_divergence_penalty: float = -0.5,
    normalize: str = "mean",   # "mean" or "median" or "max" or None
    return_plan: bool = False,
    return_details: bool = False,
    verbose: bool = False,
    sinkhorn_type: str = 'sinkhorn',  # 'knopp' or 'log'
):
    """
    fake_seq, tgt_seq: (T, C, H, W), values in [-1, 1]

    fake_seq は全フレーム使用
    tgt_seq は valid_idx のみ使用

    total = <P, M> + lambda * L_mono(P)
    を返す。

    return_plan=True:
        (total, P)
    return_details=True:
        (total, details_dict)
    両方 False:
        (total, terms_dict)
        terms_dict には少なくとも {ot_cost, reg_cost, mono_loss, mono_penalty, total} を含む。
    """
    # tgt の有効フレームだけ使う
    valid_idx = get_valid_frame_idx(tgt_seq)   # (N,)
    N = valid_idx.numel()
    T = fake_seq.shape[0]

    fake_sub = fake_seq              # (T, C, H, W)
    tgt_sub  = tgt_seq[valid_idx]    # (N, C, H, W)

    fake_flat = fake_sub.reshape(T, -1)
    tgt_flat  = tgt_sub.reshape(N, -1)

    # cost matrix: (T, N)
    M = torch.cdist(fake_flat, tgt_flat, p=2) ** 2

    # スケール正規化
    if normalize == "mean":
        s = M.detach().mean()
        M = M / (s + 1e-8)
    elif normalize == "median":
        s = M.detach().median()
        if s < 1e-6:
            s = M.detach().mean()
        M = M / (s + 1e-8)
    elif normalize == "max":
        s = M.detach().max()
        M = M / (s + 1e-8)
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unsupported normalize: {normalize}")

    # 一様分布
    a = torch.ones(T, device=fake_seq.device, dtype=fake_seq.dtype) / T
    b = torch.ones(N, device=fake_seq.device, dtype=fake_seq.dtype) / N

    
    P = ot.sinkhorn(a, b, M, reg=reg, numItermax=iters, method=sinkhorn_type)
    P = cast(torch.Tensor, P)
    OT_entropy =  - torch.sum(P * torch.log(P + 1e-8))



    # OT distance term = <P, M>
    ot_cost = torch.sum(P * M) - reg * OT_entropy
    if verbose:
        print("M min/max/mean:", M.min().item(), M.max().item(), M.mean().item())
        print("P min/max/mean:", P.min().item(), P.max().item(), P.mean().item())
        K = torch.exp(-M / reg)
        print("K min/max/mean:", K.min().item(), K.max().item(), K.mean().item())

    mono_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    U: Optional[torch.Tensor] = None

    if monotone:
        # 各 fake frame i が target 側のどの index に対応しているかの重心。
        # drop 行（row_mass≈0）は重心が 0/0 で未定義なので比較から除外し、kept 行だけを
        # 順に並べて連続ペアの逆転を測る（drop を潰さず飛ばす）。詳細は geo 版のコメント参照。
        # POT(ot.sinkhorn) は balanced なので通常は全行 kept となり従来の連番 sum と一致する。
        j_idx = torch.arange(N, device=fake_seq.device, dtype=fake_seq.dtype).view(1, N)  # (1, N)
        row_mass = P.sum(dim=1)                                                            # (T,)
        U = torch.sum(P * j_idx, dim=1) / (row_mass + 1e-8)                                # (T,)
        keep = row_mass > 0.5 * row_mass.max()                                             # (T,) kept=True/drop=False
        U_keep = U[keep]                                                                   # (K,) drop を飛ばし順序保持
        if U_keep.numel() >= 2:
            mono_loss = torch.relu(U_keep[:-1] - U_keep[1:]).sum()
        else:
            mono_loss = torch.zeros((), device=fake_seq.device, dtype=fake_seq.dtype)

    P_entropy_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    if P_entropy:
        row_mass = P.sum(dim=1, keepdim=True) + 1e-8
        Q = P / row_mass
        P_entropy_loss = -torch.sum(Q * torch.log(Q + 1e-8), dim=1)
        P_entropy_loss = P_entropy_loss.mean()

    
    ot_divergence_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    if ot_divergence:
        M_d = torch.cdist(fake_flat, fake_flat, p=2) ** 2
        b_d = torch.ones(T, device=fake_seq.device, dtype=fake_seq.dtype) / T
        P_d = ot.sinkhorn(a, b_d, M_d, reg=reg, numItermax=iters, method=sinkhorn_type)
        OT_entropy_d =  - torch.sum(P_d * torch.log(P_d + 1e-8))
        ot_divergence_loss = torch.sum(P_d * M_d) - reg * OT_entropy_d
    

        
    mono_cost = monotone_penalty * mono_loss
    entorpy_cost = P_entropy_penalty * P_entropy_loss
    ot_divergence_cost = ot_divergence_penalty * ot_divergence_loss

    total = ot_cost + mono_cost + entorpy_cost + ot_divergence_cost

    if verbose:
        print(
            "sequence_ot terms:",
            "distance=", ot_cost.item(),
            "mono=", mono_cost.item(),
            "entropy=", entorpy_cost.item(),
            "ot_divergence=", ot_divergence_cost.item(),
            "total=", total.item(),
        )

    if return_details:
        details: Dict[str, Any] = {
            "P": P,
            "M": M,
            "U": U,
            "P_d": P_d if ot_divergence else None,
            "M_d": M_d if ot_divergence else None,
            "ot_cost": ot_cost,
            "mono_cost": mono_cost,
            "entropy_cost": entorpy_cost,
            "ot_divergence_cost": ot_divergence_cost,
            "mono_loss": mono_loss,
            "mono_penalty": monotone_penalty,
            "P_entropy_loss": P_entropy_loss,
            "P_entropy_penalty": P_entropy_penalty,
            "ot_divergence_loss": ot_divergence_loss,
            "ot_divergence_penalty": ot_divergence_penalty,
            "valid_idx": valid_idx,
            "a": a,
            "b": b,
            "total": total,
        }
        return total, details

    if return_plan:
        return total, P

    terms: Dict[str, Any] = {
        "ot_cost": ot_cost,
        "mono_cost": mono_cost,
        "entropy_cost": entorpy_cost,
        "ot_divergence_cost": ot_divergence_cost,
        "mono_loss": mono_loss,
        "mono_penalty": monotone_penalty,
        "total": total,
    }
    return total, terms

def sequence_ot_loss_geo(
    fake_seq: torch.Tensor,
    tgt_seq: torch.Tensor,
    reg: float = 0.1,
    iters: int = 500,   # geomloss.ot.solve の最大反復回数
    monotone: bool = True,
    monotone_penalty: float = 50.0,
    P_entropy: bool = False,
    P_entropy_penalty: float = 1.0,
    ot_divergence: bool = False,
    ot_divergence_penalty: float = -0.5,
    normalize: Optional[str] = None,   # "mean" or "median" or "max" or None
    return_plan: bool = False,
    return_details: bool = False,
    verbose: bool = False,
    scaling: float = 0.9,   # API 互換のために残す（未使用）
    p: int = 2,             # コスト距離の次数 (cdist の p)
    unbalanced: Optional[float] = None,   # 周辺制約の KL ペナルティ ρ。None で balanced。値を渡すと unbalanced
    unbalanced_type: str = "KL",          # geomloss は現状 "KL" のみ対応
    debias: bool = False,                 # True で solve_sample(debias=True)。ot_cost を Sinkhorn divergence にする
):
    """
    geomloss.ot.solve を用いた Sinkhorn OT 実装。

    ot_cost は包絡線定理で勾配が保証される result.value を使う（value を plan へ
    再構成して微分すると勾配が壊れるため、ot_cost 用には plan を使わない）。
    一方 monotone は plan を経由してしか勾配を流せないので、非 detach の plan
    (P_grad) から重心 U を計算して勾配を流す（keep マスクの閾値判定だけは detach 済み
    plan で行い、勾配は U の値に乗る）。P_entropy は従来どおり detach 済み plan を使う。
    scaling は API 互換のために残すが未使用。iters は最大反復回数。
    戻り値の仕様は sequence_ot_loss_torch と同じ。

    unbalanced (ρ): 周辺制約の KL ソフトペナルティ強度。デフォルト None で balanced（全質量を
    必ず輸送）。値を渡したときだけ unbalanced になる。ρ が小さいほど制約が緩み、マッチしない
    余りフレームの質量を捨てられる。
    ρ は「コスト M の絶対スケール」と比べて効くので normalize=None 前提で調整すること。
    実測（reg=0.1, normalize=None, M中央値≈1.5e3, マッチ≈20-40 / 余り≈150-700）では
    ρ=60 が「良フレーム満額・余り(=T-N)フレームだけ drop」のスイートスポット。
    安定レンジは ρ∈[30,200]、ρ<20 で全行崩壊（ot_cost→0 で勾配消失）。T,N が 10〜19 で
    変動してもこのスケール感は不変（しきい値はコストで決まりフレーム本数に依存しない）。
    unbalanced_type は geomloss 現状 "KL" のみ。
    """
    from geomloss.ot import solve as geo_solve

    # tgt の有効フレームだけ使う
    valid_idx = get_valid_frame_idx(tgt_seq)   # (N,)
    N = valid_idx.numel()
    T = fake_seq.shape[0]

    fake_sub = fake_seq              # (T, C, H, W)
    tgt_sub  = tgt_seq[valid_idx]    # (N, C, H, W)

    fake_flat = fake_sub.reshape(T, -1)
    tgt_flat  = tgt_sub.reshape(N, -1)

    # 一様分布
    a = torch.ones(T, device=fake_seq.device, dtype=fake_seq.dtype) / T
    b = torch.ones(N, device=fake_seq.device, dtype=fake_seq.dtype) / N

    # debias 経路で別々にログする生コスト（loss には使わず、ログ専用 / 勾配なし）
    gen_tgt_cost: Optional[torch.Tensor] = None    # gen→tgt biased OT
    gen_gen_cost: Optional[torch.Tensor] = None    # gen→gen self OT
    divergence_cost: Optional[torch.Tensor] = None  # Sinkhorn divergence (= ot_cost when debias)

    if debias:
        # ---- Sinkhorn divergence 版 (solve_sample) ----
        # ot.solve(行列版) は固定コスト行列しか見ず gen-gen / tgt-tgt の距離を知らないので debias 不可。
        # solve_sample は点群を直接受け取り内部で self コストも作るため debias=True が使える。
        # ot_cost = res.value = OT(gen,tgt) - ½OT(gen,gen) - ½OT(tgt,tgt)（包絡線定理で微分可）。
        # 重い Sinkhorn 反復はこの1回だけ。gen→tgt / gen→gen の個別値は計算済みポテンシャルに対して
        # sinkhorn_cost の debias フラグを変えて呼び直すだけで取れる（追加 solve 不要・ログ専用 → detach）。
        # コストは sqeuclidean 固定（= cdist(...,p=2)**2 相当）。normalize は点群版では適用しない。
        from geomloss.ot import solve_sample as geo_solve_sample
        from geomloss.ot._abstract_solvers.unbalanced_ot import sinkhorn_cost as _sinkhorn_cost

        res = geo_solve_sample(
            fake_flat, tgt_flat, reg=reg, a=a, b=b,
            cost="sqeuclidean", debias=True, max_iter=iters,
            unbalanced=unbalanced, unbalanced_type=unbalanced_type,
        )
        P_grad = cast(torch.Tensor, res.plan)   # gen→tgt の輸送計画（monotone 用に勾配保持）
        P = P_grad.detach()
        ot_cost = cast(torch.Tensor, res.value)  # = Sinkhorn divergence（最適化する損失項）
        # 点群版は内部でコスト行列を作らないが、スナップショット可視化用に return_details 時だけ
        # gen→tgt の sqeuclidean コスト行列（= solve_sample 内部コスト ‖x−y‖²）を作って詰める。
        # ログ専用なので detach。学習ステップ（return_details=False）では作らない。
        M = (torch.cdist(fake_flat, tgt_flat, p=2) ** 2).detach() if return_details else None

        # ログ専用の個別値（勾配不要 → no_grad / detach）
        with torch.no_grad():
            pot = res._potentials               # SinkhornPotentials(g_ab, f_ba, f_aa, g_bb)
            SP = type(pot)
            # フラグ反転: 同じポテンシャルで debias=False → gen→tgt biased OT
            gen_tgt_cost = _sinkhorn_cost(
                a=a, b=b, potentials=pot, eps=reg,
                rho=unbalanced, debias=False, batchsize=0,
            ).detach()
            # gen→gen self: f_aa を (gen,gen) 問題の両ポテンシャルとして biased 評価
            # （balanced/unbalanced とも厳密。2⟨a,f_aa⟩ は balanced 限定なので使わない）
            self_pot = SP(g_ab=pot.f_aa, f_ba=pot.f_aa, f_aa=None, g_bb=None)
            gen_gen_cost = _sinkhorn_cost(
                a=a, b=a, potentials=self_pot, eps=reg,
                rho=unbalanced, debias=False, batchsize=0,
            ).detach()
            divergence_cost = ot_cost.detach()
    else:
        # ---- 従来の行列版 (biased OT, ot.solve) ----
        # cost matrix: (T, N)
        M = torch.cdist(fake_flat, tgt_flat, p=p) ** p

        # スケール正規化
        if normalize == "mean":
            s = M.detach().mean() + 1e-8
            M = M / s
        elif normalize == "median":
            s = M.detach().median()
            if s < 1e-6:
                s = M.detach().mean()
            M = M / (s + 1e-8)
        elif normalize == "max":
            s = M.detach().max() + 1e-8
            M = M / s
        elif normalize is None:
            pass
        else:
            raise ValueError(f"Unsupported normalize: {normalize}")

        # geomloss.ot.solve で OT を解く。
        # 勾配は包絡線定理で微分可能性が保証される result.value から取る。
        # plan を再構成して微分すると勾配が壊れる（実測で約 1000 倍に爆発）ので、
        # plan は detach し monotone/P_entropy の値計算と return_plan 用にのみ使う。
        result = geo_solve(
            M, reg=reg, a=a, b=b, max_iter=iters,
            unbalanced=unbalanced, unbalanced_type=unbalanced_type,
        )
        # plan は (1) ot_cost は envelope theorem で微分可能な result.value から取るので不要だが、
        # (2) monotone は plan を経由してしか勾配を流せないので非 detach の plan を保持する。
        # P_grad: 勾配を流す用（monotone）。P: detach 済みで値計算・P_entropy・return 用。
        P_grad = cast(torch.Tensor, result.plan)
        P = P_grad.detach()

        # OT distance term = entropic OT value（biased）
        ot_cost = cast(torch.Tensor, result.value)

    if verbose:
        if M is not None:
            print("M min/max/mean:", M.min().item(), M.max().item(), M.mean().item())
        print("P min/max/mean:", P.min().item(), P.max().item(), P.mean().item())
        print("P row sums:", P.sum(dim=1).detach().cpu().numpy())

    mono_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    U: Optional[torch.Tensor] = None

    if monotone:
        # 各 fake frame i が target 側のどの index に対応しているかの重心（条件付き）。
        # 注: geo 版でも monotone は P_grad（非 detach の plan）から計算し、plan 経由で勾配を流す。
        # unbalanced OT は質量を連続的に緩和するので、ほとんど輸送されない (row_mass≈0) drop 行が
        # 出る。drop 行は U = Σ(P·j)/row_mass が 0/0 で重心が定義できず、単調性の比較に入れては
        # いけない。そこで (1) 質量が十分ある kept 行だけを残し、(2) drop を「飛ばして」連続する
        # kept 行同士で重心の逆転 relu(U_prev - U_next) を測る。
        # drop を含むペアを 0 で潰す方式だと、drop を挟んだ kept ペア間の逆転が検出できなくなる
        # （例: 1 つおき drop だと全ペアが 0 になり mono_loss が消える）ため、潰すのではなく飛ばす。
        # kept 判定は一様質量 a(=1/T) に対する絶対閾値。相対閾値(max の半分)だと全質量ドロップ時
        # (row_mass が全行 ~1e-16 で横並び)でも半数が kept 扱いになり、その行の U が 1/(row_mass+eps)
        # ≈ 1/eps で勾配爆発する。絶対閾値なら全ドロップ時は keep が空 → mono_loss=0 となり、
        # 「質量が流れていない＝重心は無意味なので寄与 0」という正しい挙動になる。
        # balanced では row_mass≈a で全行 kept なので従来の連番 sum と一致する。
        # さらに分母 row_mass は detach する: 値は P_grad から作るが、1/(row_mass) の勾配増幅
        # (row_mass≈0 で 1/eps 倍) を断ち、勾配は分子(輸送先 index の重心)経由だけに流す。
        # keep マスク（行選択の閾値）は勾配不要なので detach 済み P で決める。
        j_idx = torch.arange(N, device=fake_seq.device, dtype=fake_seq.dtype).view(1, N)  # (1, N)
        U = torch.sum(P_grad * j_idx, dim=1) / (P_grad.sum(dim=1).detach() + 1e-8)         # (T,) 分子のみ勾配
        row_mass_det = P.sum(dim=1)                                                        # (T,) 閾値判定用(detach)
        keep = row_mass_det > 0.5 * a                                                      # (T,) kept=True/drop=False
        U_keep = U[keep]                                                                   # (K,) drop を飛ばし順序保持
        if U_keep.numel() >= 2:
            mono_loss = torch.relu(U_keep[:-1] - U_keep[1:]).sum()
        else:
            mono_loss = torch.zeros((), device=fake_seq.device, dtype=fake_seq.dtype)

    P_entropy_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    if P_entropy:
        # unbalanced で drop 行(row_mass≈0)は条件付き分布 Q が不定になるため、
        # 各行を rel∈[0,1] (本来質量に対する割合) で重み付けした加重平均にする。
        # balanced では rel≈1 となり従来の単純平均にほぼ一致。
        row_mass = P.sum(dim=1, keepdim=True)                                              # (T,1)
        Q = P / (row_mass + 1e-8)
        row_ent = -torch.sum(Q * torch.log(Q + 1e-8), dim=1)                               # (T,)
        w = torch.clamp(row_mass.squeeze(1) / (a + 1e-12), max=1.0)                        # (T,) kept≈1/drop≈0
        P_entropy_loss = (w * row_ent).sum() / (w.sum() + 1e-8)

    ot_divergence_loss = torch.tensor(0.0, device=fake_seq.device, dtype=fake_seq.dtype)
    P_d: Optional[torch.Tensor] = None
    M_d: Optional[torch.Tensor] = None
    if ot_divergence and not debias:
        # 注意: fake-fake の生 OT (biased)。最適計画が恒等行列に退化し値・勾配とも
        # ほぼ 0 になる。SamplesLoss 相当の Sinkhorn divergence が欲しい場合は debias=True を使う
        # （ot_cost 自体が divergence になり、この手作り項は不要なのでスキップする）。
        M_d = torch.cdist(fake_flat, fake_flat, p=p) ** p
        b_d = torch.ones(T, device=fake_seq.device, dtype=fake_seq.dtype) / T
        result_d = geo_solve(M_d, reg=reg, a=a, b=b_d, max_iter=iters)
        P_d = cast(torch.Tensor, result_d.plan).detach()
        ot_divergence_loss = cast(torch.Tensor, result_d.value)

    mono_cost = monotone_penalty * mono_loss
    entorpy_cost = P_entropy_penalty * P_entropy_loss
    ot_divergence_cost = ot_divergence_penalty * ot_divergence_loss

    total = ot_cost + mono_cost + entorpy_cost + ot_divergence_cost

    if verbose:
        print(
            "sequence_ot(geo) terms:",
            "distance=", ot_cost.item(),
            "mono=", mono_cost.item(),
            "entropy=", entorpy_cost.item(),
            "ot_divergence=", ot_divergence_cost.item(),
            "total=", total.item(),
        )

    if return_details:
        details: Dict[str, Any] = {
            "P": P,
            "M": M,
            "U": U,
            "P_d": P_d if ot_divergence else None,
            "M_d": M_d if ot_divergence else None,
            "ot_cost": ot_cost,
            "mono_cost": mono_cost,
            "entropy_cost": entorpy_cost,
            "ot_divergence_cost": ot_divergence_cost,
            "mono_loss": mono_loss,
            "mono_penalty": monotone_penalty,
            "P_entropy_loss": P_entropy_loss,
            "P_entropy_penalty": P_entropy_penalty,
            "ot_divergence_loss": ot_divergence_loss,
            "ot_divergence_penalty": ot_divergence_penalty,
            "unbalanced": unbalanced,
            "unbalanced_type": unbalanced_type,
            "debias": debias,
            "gen_tgt_cost": gen_tgt_cost,        # debias 時のみ: gen→tgt biased OT (log用)
            "gen_gen_cost": gen_gen_cost,        # debias 時のみ: gen→gen self OT  (log用)
            "divergence_cost": divergence_cost,  # debias 時のみ: Sinkhorn divergence (= ot_cost)
            "transported_mass": float(P.sum()),  # balanced なら≈1, unbalanced で N/T 付近へ
            "valid_idx": valid_idx,
            "a": a,
            "b": b,
            "total": total,
        }
        return total, details

    if return_plan:
        return total, P

    terms: Dict[str, Any] = {
        "ot_cost": ot_cost,
        "mono_cost": mono_cost,
        "entropy_cost": entorpy_cost,
        "ot_divergence_cost": ot_divergence_cost,
        "mono_loss": mono_loss,
        "mono_penalty": monotone_penalty,
        "gen_tgt_cost": gen_tgt_cost,        # debias 時のみ: gen→tgt biased OT (log用)
        "gen_gen_cost": gen_gen_cost,        # debias 時のみ: gen→gen self OT  (log用)
        "divergence_cost": divergence_cost,  # debias 時のみ: Sinkhorn divergence (= ot_cost)
        "total": total,
    }
    return total, terms

def sequence_ot_loss(
    fake_seq: torch.Tensor,
    tgt_seq: torch.Tensor,
    solver: str = "pot",            # "pot" (POT/ot.sinkhorn) or "geo" (GeomLoss)
    reg: float = 0.1,
    iters: int = 500,               # POT: numItermax / geo: max_iter
    monotone: bool = True,
    monotone_penalty: float = 50.0,
    P_entropy: bool = False,
    P_entropy_penalty: float = 1.0,
    ot_divergence: bool = False,
    ot_divergence_penalty: float = -0.5,
    normalize: Optional[str] = None,   # unbalanced(ρ) は raw コストスケール前提なので geo は None
    return_plan: bool = False,
    return_details: bool = False,
    verbose: bool = False,
    sinkhorn_type: str = "sinkhorn",  # POT 専用
    geo_scaling: float = 0.99,        # geo 専用 (ε-scaling の細かさ)
    geo_p: int = 2,                   # geo 専用 (コスト次数)
    unbalanced: Optional[float] = None,  # geo 専用 (周辺制約 KL ペナルティ ρ。None で balanced。値を渡すと unbalanced)
    unbalanced_type: str = "KL",         # geo 専用 (現状 "KL" のみ)
    debias: bool = False,                # geo 専用 (True で solve_sample(debias=True)。ot_cost を Sinkhorn divergence にする)
):
    """
    POT版 (sequence_ot_loss_torch) と GeomLoss版 (sequence_ot_loss_geo) を
    solver で切り替えるディスパッチャ。共通の引数だけ受け取り、solver 固有の
    引数 (POT: iters/sinkhorn_type, geo: geo_scaling/geo_p/unbalanced/
    unbalanced_type) は該当する実装にのみ渡す。戻り値の仕様は両実装で共通。
    """
    common = dict(
        reg=reg,
        monotone=monotone,
        monotone_penalty=monotone_penalty,
        P_entropy=P_entropy,
        P_entropy_penalty=P_entropy_penalty,
        ot_divergence=ot_divergence,
        ot_divergence_penalty=ot_divergence_penalty,
        normalize=normalize,
        return_plan=return_plan,
        return_details=return_details,
        verbose=verbose,
    )
    if solver == "geo":
        # geo は iters/sinkhorn_type を取らない (ε-scaling は geo_scaling で制御)
        return sequence_ot_loss_geo(
            fake_seq, tgt_seq,
            scaling=geo_scaling,
            p=geo_p,
            iters=iters,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            debias=debias,
            **common,
        )
    elif solver == "pot":
        return sequence_ot_loss_torch(
            fake_seq, tgt_seq,
            iters=iters,
            sinkhorn_type=sinkhorn_type,
            **common,
        )
    else:
        raise ValueError(f"Unsupported solver: {solver!r} (use 'pot' or 'geo')")
