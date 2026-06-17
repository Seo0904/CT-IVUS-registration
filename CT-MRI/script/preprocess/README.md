# SynthRAD2023 CT→MRI 生成用 前処理パイプライン

CT を入力（条件）、MRI を生成ターゲットとする image synthesis 向けに、
SynthRAD2023 の配布済み NIfTI（rigid位置合わせ・輪郭マスク・defacing・クロップ・
リサンプリングは公式パイプラインで完了済み）を、学習可能な正規化済みデータへ変換する。

```
<root>/<region>/<patientID>/mr.nii.gz   # MR (target)
                           /ct.nii.gz   # CT (input/condition)
                           /mask.nii.gz # 患者輪郭マスク
```

## なぜ CT と MR で正規化を分けるか（設計の核）

CT と MR は画素値の性質が真逆なので、正規化の思想を変える。

| | CT（入力） | MR（ターゲット） |
|---|---|---|
| 値の意味 | **物理量 HU**（絶対値・症例間で一貫） | **任意単位**（スキャナ/シーケンス依存、症例間でバラバラ） |
| 方針 | **固定HU窓 → 線形 [-1,1]** | **N4 → Nyúl(train限定fit) → マスク内 [-1,1]** |
| 禁止 | 症例ごと percentile / z-score（絶対値整合を壊す） | landmark を全データで fit（リーク） |

固定HU窓は実データで決定した（決め打ちしない）：`script/preprocess/analyze_hu.py` で
brain/pelvis のマスク内HU分布を確認し、骨 p99.9（brain≈1540 / pelvis≈1240）を完全に
内包する **`[-1000, 2000]`** を採用。下限-1000は両領域の空気ピーク、>2000の金属/再構成
飽和値のみクリップされ情報損失はほぼ無い。

## パイプライン（2パス構成）

Nyúl の「train限定 fit」を正しく分けるため 2パスにしている。

```
Pass1 : 全症例   空間整合(assert/必要時resample) + 背景固定 + N4(MR)  → _cache/
Fit   : trainのみ Nyúl landmark を fit（val/testは渡さない＝リーク防止） → nyul_standard_histogram.npy
Pass2 : 全症例   CT窓化[-1,1] / MR(Nyúl適用 + マスク内[-1,1]) を確定   → 出力
```

### 各ステップの根拠

**空間系（CT・MR共通）**
1. `same_grid()` で CT=MR=mask の size/spacing/origin/direction を検証（assert相当）。
   ズレていれば CT=線形 / mask=最近傍 で **MRグリッドへ resample**（ペア対応を壊さない）。
   ※ SynthRAD配布データは既に同一グリッドなので実際には素通りする（`grid=identical`）。
2. **背景（マスク外）を固定**：CT は `bg_ct_hu=-1000`（窓化で -1 に張り付く）。
   以降の統計は**必ずマスク内のみ**（背景空気を混ぜない）。
3. MR/CT で**別々のリサンプリング・別々のクロップをしない**（常にMR基準）。

**CT 強度**
- `clip([-1000,2000]) → 2*(x-lo)/(hi-lo)-1`。症例ごと percentile/z-score は使わない。
- `two_channel=true` で広窓＋軟部窓 `[-160,240]` の 2ch 出力（軟部コントラスト強調、特に pelvis）。

**MR 強度**
- **N4 bias field correction**（SimpleITK、shrinkで高速化し full-res にバイアス場適用）。
- **Nyúl histogram matching**（`intensity-normalization`）。**landmark は train のみで fit**、
  val/test は適用するだけ。標準ヒストグラムを保存し再現/適用可能に。
- 前景は `mask AND (mr>0)`。defacing 等でマスク内に多数ある「正確に0」のボクセルを除外
  （含めると low percentile が0重複→Nyúl補間がゼロ割れ→NaN になるため）。ゼロ領域は -1。
- 仕上げに **マスク内 [-1,1] スケール**（既定、p0.5..p99.5でロバスト）または **マスク内 z-score**。

**仕上げ**
- `out_size` 指定時はアスペクト比保持リサイズ＋背景パディング（in-plane のみ、z不変）。
- ターゲットMRを生成器出力レンジ `[-1,1]` に合わせる。

> 注: brain と pelvis は MR 強度スケールが桁違い（brain 0〜3000 / pelvis 0〜数百）なので、
> **Nyúl は region ごとに別 fit すること**（`--region` を分けて実行）。

## 使い方

```bash
cd CT-MRI/script/preprocess

# 0) [推奨] 固定HU窓の妥当性を実データで確認
python analyze_hu.py --n 6                 # brain/pelvis のHU/spacing/サイズを出力

# 1) 本処理
python run.py --config config.yaml --region brain
python run.py --config config.yaml --region pelvis

# オプション
python run.py --region pelvis --two_channel          # CTを2ch化
python run.py --region brain  --mr_norm zscore       # MR仕上げをz-score
python run.py --region brain  --limit 8              # 先頭8症例だけ(デバッグ)
python run.py --region brain  --skip_pass1           # N4キャッシュ済みなら再利用
```

設定はすべて `config.yaml`（HU窓・2ch・MR標準化方式・out_size・分割・パス）。
CLI 引数は config を上書きする。

## 出力

```
<out_root>/<region>/
├── <region>.h5                 # 学習用: groups CT / MR / MASK, 各 patientID dataset, attrs[split,affine]
├── nifti/<cid>/                # 確認用: ct(.nii.gz|ct_ch0/1), mr, mask（affine保持）
├── params/<cid>.json           # 逆変換用: HU窓, MR統計(lo/hi or mean/std), affine, nyul履歴
├── qc/<cid>_overlay.png        # 中央 axial/coronal/sagittal の CT/MR + mask輪郭
├── qc/<cid>_hist.png           # 処理前後ヒストグラム（CT生HU / MR生 / MR正規化後）
├── nyul_standard_histogram.npy # train-fit済み Nyúl landmark
├── split.json / grid_info.json / config_snapshot.json
└── _cache/<cid>/               # 中間(ct_hu/mr_n4/mask)。再実行高速化用
```

## やってはいけないこと（このコードで回避済み）
- 2Dスライス単位の正規化（3D連続性破壊）→ 全処理は3Dボリューム単位。
- CT の percentile / z-score → 固定HU窓のみ。
- 背景空気を含めた統計 → 統計は常にマスク内（MRはさらにゼロ除外）。
- MR/CT で別リサンプル・別クロップ → 常にMR基準で同一処理。
- Nyúl を全データで fit → train のみ fit、val/test は適用のみ。
- N4 の省略 → 必須ステップとして Pass1 で実行。

## 依存
`SimpleITK`, `intensity-normalization`(v3), `numpy`, `nibabel`, `h5py`, `scipy`, `PyYAML`, `matplotlib`
