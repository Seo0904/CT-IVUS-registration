"""config.yaml を型付き dataclass に読み込むローダ。

CLI から一部フィールド(region, limit, out_root, two_channel など)を上書きできる。
ネストした節(n4/nyul/split)は専用 dataclass に展開する。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any, Dict
import yaml


@dataclass
class N4Config:
    shrink_factor: int = 4
    n_iterations: List[int] = field(default_factory=lambda: [50, 50, 50, 50])


@dataclass
class NyulConfig:
    output_min: float = 0.0
    output_max: float = 100.0
    min_percentile: float = 1.0
    max_percentile: float = 99.0
    percentile_step: float = 10.0
    max_fit_images: int = 40


@dataclass
class SplitConfig:
    mode: str = "ratio"                 # ratio | explicit
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    seed: int = 0
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)


@dataclass
class Config:
    # 入出力
    region: str = "brain"
    data_root: str = "/workspace/data/org_data/synthRAD"
    out_root: str = "/workspace/data/dust_box/CT-MRI/preprocessed"
    limit: Optional[int] = None

    # 空間系
    grid_atol: float = 1e-3
    resample_if_mismatch: bool = True
    bg_ct_hu: float = -1000.0

    # CT
    hu_window: Tuple[float, float] = (-1000.0, 2000.0)
    two_channel: bool = False
    soft_window: Tuple[float, float] = (-160.0, 240.0)

    # MR
    n4: N4Config = field(default_factory=N4Config)
    nyul: NyulConfig = field(default_factory=NyulConfig)
    mr_norm: str = "minmax"             # minmax | zscore
    clip_zscore: float = 3.0

    # 仕上げ
    out_size: Optional[Tuple[int, int]] = None
    out_formats: List[str] = field(default_factory=lambda: ["nifti", "h5"])

    # 真っ黒スライス除去 (z軸): CT/MRどちらかが前景比率 <= しきい値 のスライスをペアごと除去
    drop_black_frames: bool = True
    black_fg_ratio: float = 0.02        # 前景比率(=前景画素/面内画素)がこれ以下なら真っ黒扱い
                                        # 2% = 128²で~327px。MR端の薄い三日月エッジまで除去

    # 分割
    split: SplitConfig = field(default_factory=SplitConfig)

    # QC
    qc: bool = True

    # --- 派生 ---
    @property
    def region_dir(self) -> str:
        import os
        return os.path.join(self.data_root, self.region)

    @property
    def out_dir(self) -> str:
        import os
        return os.path.join(self.out_root, self.region)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                raw[k] = v

    n4 = N4Config(**(raw.pop("n4", {}) or {}))
    nyul = NyulConfig(**(raw.pop("nyul", {}) or {}))
    split = SplitConfig(**(raw.pop("split", {}) or {}))

    # タプル化(yaml はリストで来る)
    for key in ("hu_window", "soft_window", "out_size"):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    if raw.get("out_size") is not None:
        raw["out_size"] = tuple(int(x) for x in raw["out_size"])

    cfg = Config(n4=n4, nyul=nyul, split=split, **raw)
    return cfg
