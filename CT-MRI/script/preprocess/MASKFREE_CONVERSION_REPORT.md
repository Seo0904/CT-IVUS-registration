# マスク不使用 (mask-free) 前処理への変換レポート

SynthRAD2023 brain CT→MR 前処理を、**マスク(`mask.nii.gz`/GT segmentation)に一切依存しない**形へ変換した。対応(レジストレーション)の無い生データに適用する手法という前提に合わせ、正規化を強度ベースに置き換えた。実装は [`run_maskfree.py`](run_maskfree.py)。

- 入力: `/workspace/data/org_data/synthRAD/brain/<case>/{ct,mr}.nii.gz`（`mask.nii.gz` は**読まない**）
- 出力: `/workspace/data/preprocessed/synthRAD_maskfree/brain/brain.h5`（群は `CT`, `MR` のみ。`MASK` 無し）

---

## 1. 動機

従来の [`run.py`](run.py) / [`pipeline.py`](pipeline.py) は、出力 CT/MR の**画素値そのもの**をマスクで決めていた。マスクが無い生データには再現できず、「対応の無いデータに使う手法」という建付けと矛盾する。そこでマスク依存を全て除去した。

## 2. 元のマスク依存箇所と置換方法

| 元の処理 | 箇所 | マスク依存の中身 | 置換(mask-free) |
|---|---|---|---|
| CT 背景固定 | [pipeline.py:107-109](pipeline.py#L107) | マスク外を空気HU(-1000)に塗る | **廃止**。CTのHUは絶対量で体外は元から空気。**HU窓 [-1000,2000]→[-1,1] のみ**（[window_to_unit](pipeline.py#L95)） |
| MR スケール統計 | [pipeline.py:205-216](pipeline.py#L205) | `[-1,1]` の percentile/mean をマスク内で計算、背景→-1 | 統計領域を **Otsu強度前景**に置換（同関数を前景マスク引数で再利用） |
| N4 領域 | [pipeline.py:136](pipeline.py#L136) | バイアス推定をマスク内に限定 | **Otsu前景**を N4 のマスクに使用 |
| Nyul 前景 | [pipeline.py:168](pipeline.py#L168) | `(mask>0.5)&(mr>0)` | `(Otsu前景>0.5)&(mr>0)` に置換 |
| mask 運搬/保存/QC | [run.py](run.py) 各所 | mask の resample・保存・赤枠QC | **削除**（h5 に MASK を保存しない） |

### Otsu 強度前景（マスクの代替）
[`otsu_foreground_arr`](run_maskfree.py) — MR 自身から体領域を推定:
1. 非ゼロ画素に Otsu 閾値（背景=空気/defacingのゼロ、前景=組織は明るい、を利用）
2. `mr > thr` を二値化 → `binary_fill_holes` → **最大連結成分**を体として採用

GT マスクの代わりにこの前景を N4 / Nyul / MRスケール統計の領域に使う。**CT 側は前景推定すら不要**（絶対HU窓のみ）。

## 3. CT / MR の正規化（確定仕様）

- **CT**: `clip(HU, -1000, 2000)` を線形に `[-1,1]`。背景処理なし・症例間で同一変換（HUの絶対整合を保持）。
- **MR**: `N4(Otsu前景)` → `Nyul`（**train症例のみ**で fit, リーク防止, Otsu前景使用）→ `Otsu前景内 minmax(p0.5–p99.5)` で `[-1,1]`、前景外は `-1`。

## 4. 本変換で追加した仕様

| 項目 | 値 | 実装 |
|---|---|---|
| in-plane リサイズ | **128×128**（アスペクト比保持・レターボックス、bg=-1） | [resize_keep_aspect](pipeline.py#L235) を out_size=(128,128) で適用 |
| フレーム間引き | **1枚おき（半分）** `vol[:,:,::2]`（axial=z軸） | [run_maskfree.py pass2](run_maskfree.py) |
| volume 数 | **先頭 90**（case-id ソート順, brain） | `--num_volumes 90` |
| split | ratio 0.7/0.1/0.2, seed 0 を 90 例に再生成 → train 63 / val 9 / test 18 | [make_split](run_maskfree.py) |

正規化はネイティブ解像度で行い、**その後**に 128 リサイズ→フレーム間引きを実施（順序: 正規化 → resize → `[::2]`）。

## 5. 出力構造

```
synthRAD_maskfree/brain/
├── brain.h5                       # 群: CT, MR のみ（各 case_id -> (128,128,~100) float32, [-1,1]）
├── split.json                     # case_id -> train/val/test
├── nyul_standard_histogram.npy    # Nyul landmark（train-fit）
├── config_maskfree.json           # 変換設定スナップショット
├── grid_info.json
└── _cache/                        # 中間(align/N4/Otsu前景)。再実行用、学習には不要
```

検証（先頭3例の試走）:
- CT ∈ [-1, ~0.9–1.0]、MR ∈ [-1, 1]、`MASK` 群なし
- Otsu前景率 ≈ 0.17、MR 前景内 mean ≈ -0.1（つぶれていない）
- 形状 (128,128,~100)（z: 204→102 等、半減を確認）
- 視覚確認: CT 頭蓋・MR 脳実質が保たれ、ペアで整合

## 6. 学習側の対応

[`WP-UNSB_ct_mri/data/ct_mri_dataset.py`](../../../WP-UNSB_ct_mri/data/ct_mri_dataset.py) を更新:
- **`MASK` 群が無ければ全スライスを使用**（`min_mask_ratio` によるマスク依存のスライス選択を回避）。
- フレーム間引きは前処理で焼き込み済みなので、loader は格納スライスをそのまま 1 sequence として使う。

使用例:
```bash
python train.py --dataroot /workspace/data/preprocessed/synthRAD_maskfree \
  --dataset_mode ct_mri --region brain --batch_size 1 ...
```

## 7. 注意・限界

- **Otsu前景 ≠ GTマスク**。境界はラフで、低信号組織を取りこぼす／空気を含むことがある。あくまで**正規化の領域指定**用途であり、GT精度は不要。評価でマスクが要る場合は別途用意すること。
- 128 リサイズは**不可逆な解像度低下**を伴う（医療画像の微細構造は失われる）。本データは高速な開発・反復用。最終評価にはネイティブ解像度を別途検討。
- フレーム半引き・先頭90 volume はデータ削減のための仕様であり、フル設定とは分布が異なる。
