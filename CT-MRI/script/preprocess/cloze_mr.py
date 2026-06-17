"""
CT->MRI maskfree h5 の MR を「ランダムに真っ黒スライス化」する cloze 変換。

MovingMNIST の cloze.py(各サンプルで n_remove フレームを 0 に置換)と同じ発想だが、
本データは [-1,1] 正規化で背景=-1 が真っ黒なので、置換値は 0 ではなく -1.0(=黒)。

仕様:
  - 入力 h5 の CT はそのままコピー、MR のみ各 volume で n_remove 枚を黒(-1)に置換。
  - n_remove は [min_remove, max_remove] から volume ごとにランダム(最低1枚は残す)。
  - frame 軸は配列の最終軸(z)。属性(split/affine)は引き継ぐ。
  - どの z を消したかを各 MR dataset の属性 cloze_removed / cloze_n_remove に記録(再現用)。

使い方:
  python cloze_mr.py \
    --input  /workspace/data/preprocessed/synthRAD_maskfree_full/brain/brain.h5 \
    --output /workspace/data/preprocessed/synthRAD_maskfree_full/brain/brain_cloze.h5 \
    --min_remove 10 --max_remove 30 --seed 42
"""
from __future__ import annotations

import argparse
import numpy as np
import h5py


def get_args():
    p = argparse.ArgumentParser(description="MR random black-slice cloze for maskfree h5")
    p.add_argument("--input", type=str,
                   default="/workspace/data/preprocessed/synthRAD_maskfree_full/brain/brain.h5")
    p.add_argument("--output", type=str,
                   default="/workspace/data/preprocessed/synthRAD_maskfree_full/brain/brain_cloze.h5")
    p.add_argument("--min_remove", type=int, default=10)
    p.add_argument("--max_remove", type=int, default=30)
    p.add_argument("--black_value", type=float, default=-1.0,
                   help="真っ黒の値([-1,1]正規化の背景=-1)")
    p.add_argument("--target_group", type=str, default="MR",
                   help="cloze を適用するグループ(既定 MR)。他グループはそのままコピー")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        groups = list(fin.keys())
        # cloze 対象以外のグループは丸ごとコピー(CT など)
        for g in groups:
            if g == args.target_group:
                continue
            fin.copy(g, fout)

        # 対象グループ(MR)を volume ごとに cloze
        gin = fin[args.target_group]
        gout = fout.create_group(args.target_group)
        keys = sorted(gin.keys())

        tot_slices = tot_removed = 0
        for cid in keys:
            ds = gin[cid]
            arr = ds[...]                     # (H, W, D)
            D = arr.shape[-1]

            n_remove = int(rng.integers(args.min_remove, args.max_remove + 1))
            n_remove = min(n_remove, D - 1)   # 最低1枚は残す
            remove_z = np.sort(rng.choice(D, size=n_remove, replace=False))

            arr[..., remove_z] = args.black_value

            d = gout.create_dataset(cid, data=arr, compression="gzip")
            # 元の属性(split/affine)を引き継ぐ
            for k, v in ds.attrs.items():
                d.attrs[k] = v
            d.attrs["cloze_removed"] = remove_z.astype(np.int32)
            d.attrs["cloze_n_remove"] = n_remove

            tot_slices += D
            tot_removed += n_remove

        # ルート属性にcloze設定を記録
        fout.attrs["cloze_target"] = args.target_group
        fout.attrs["cloze_min_remove"] = args.min_remove
        fout.attrs["cloze_max_remove"] = args.max_remove
        fout.attrs["cloze_black_value"] = args.black_value
        fout.attrs["cloze_seed"] = args.seed

    print(f"saved: {args.output}")
    print(f"  volumes={len(keys)}  groups={groups}  target={args.target_group}")
    print(f"  removed {tot_removed}/{tot_slices} slices "
          f"({100*tot_removed/tot_slices:.1f}%)  black={args.black_value} "
          f"range=[{args.min_remove},{args.max_remove}]/vol")


if __name__ == "__main__":
    main()
