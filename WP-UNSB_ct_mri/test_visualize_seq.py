"""
テスト結果のシーケンス可視化スクリプト。

fake_5 / real / real_B の各フレームをシーケンスごとに横に並べて
1枚の画像（3行: real_A / fake_5 / real_B）として保存する。

使用方法:
    python3 test_visualize_seq.py <images_dir> [--out_dir OUT_DIR] [--max_seq N]

例:
    python3 test_visualize_seq.py \
        /path/to/test_results/.../test_best/images \
        --out_dir /path/to/test_results/.../test_best/seq_vis
"""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CATEGORIES = [
    ("real",   "real_A"),
    ("real_B", "real_B"),
    ("fake_5", "fake_5"),
]

LABEL_WIDTH = 80   # px — left column for row labels
PAD = 4            # px — gap between frames


def load_seq_frames(folder: Path, seq_id: int) -> list[np.ndarray]:
    """seqNのフレームをframe番号順に読み込む。"""
    pattern = re.compile(rf"^seq{seq_id}_frame_(\d+)\.png$")
    entries = []
    for p in folder.iterdir():
        m = pattern.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda x: x[0])
    return [np.array(Image.open(p).convert("L")) for _, p in entries]


def make_strip(frames: list[np.ndarray], pad: int = PAD) -> np.ndarray:
    """フレームリストを横方向に連結してストリップ画像を返す。"""
    h, w = frames[0].shape
    total_w = len(frames) * w + (len(frames) - 1) * pad
    strip = np.zeros((h, total_w), dtype=np.uint8)
    for i, f in enumerate(frames):
        x = i * (w + pad)
        strip[:, x:x + w] = f
    return strip


def make_label_col(row_h: int, label: str, label_w: int) -> np.ndarray:
    """左端のラベル列（縦書きテキスト）を生成する。"""
    col = np.zeros((row_h, label_w), dtype=np.uint8)
    img = Image.fromarray(col)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((label_w - tw) // 2, (row_h - th) // 2), label, fill=200, font=font)
    return np.array(img)


def visualize_seq(images_dir: Path, seq_id: int) -> Image.Image:
    """1シーケンス分の3行比較画像を生成する。"""
    rows = []
    for folder_name, label in CATEGORIES:
        folder = images_dir / folder_name
        if not folder.exists():
            continue
        frames = load_seq_frames(folder, seq_id)
        if not frames:
            continue
        strip = make_strip(frames)
        label_col = make_label_col(strip.shape[0], label, LABEL_WIDTH)
        row = np.concatenate([label_col, strip], axis=1)
        rows.append(row)

    if not rows:
        raise ValueError(f"seq{seq_id} のフレームが見つかりません")

    # タイトル行
    h_each, total_w = rows[0].shape
    title_h = 18
    canvas_h = title_h + len(rows) * h_each + (len(rows) - 1) * PAD
    canvas = np.zeros((canvas_h, total_w), dtype=np.uint8)

    title_img = Image.fromarray(np.zeros((title_h, total_w), dtype=np.uint8))
    draw = ImageDraw.Draw(title_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    title_text = f"seq {seq_id}"
    bbox = draw.textbbox((0, 0), title_text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 2), title_text, fill=220, font=font)
    canvas[:title_h] = np.array(title_img)

    y = title_h
    for i, row in enumerate(rows):
        canvas[y:y + h_each] = row
        y += h_each + PAD

    return Image.fromarray(canvas)


def main():
    parser = argparse.ArgumentParser(description="テスト結果シーケンス可視化")
    parser.add_argument("images_dir", type=Path, nargs='?',
                         default="/workspace/data/experiment_result/WP-UNSB_min_gan/moving-mnist/20260607_041405/test_results/moving_mnist_seg_paired_sb_wo_GL_w_otdiv_015_cloze_geo_gan/test_best/images"
                        ,help="test結果のimagesディレクトリ (fake_5/real/real_B を含むフォルダ)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="出力先ディレクトリ (デフォルト: images_dir/../seq_vis)")
    parser.add_argument("--max_seq", type=int, default=None,
                        help="可視化するシーケンス数の上限 (デフォルト: 全て)")
    args = parser.parse_args()

    images_dir: Path = args.images_dir.resolve()
    out_dir: Path = args.out_dir or images_dir.parent / "seq_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 利用可能なシーケンスIDを収集
    ref_folder = images_dir / "fake_5"
    if not ref_folder.exists():
        ref_folder = images_dir / "real"
    seq_ids = sorted({
        int(m.group(1))
        for p in ref_folder.iterdir()
        if (m := re.match(r"^seq(\d+)_frame_\d+\.png$", p.name))
    })

    if args.max_seq is not None:
        seq_ids = seq_ids[:args.max_seq]

    print(f"images_dir : {images_dir}")
    print(f"out_dir    : {out_dir}")
    print(f"sequences  : {len(seq_ids)}")

    for seq_id in seq_ids:
        img = visualize_seq(images_dir, seq_id)
        out_path = out_dir / f"test_seq{seq_id:04d}.png"
        img.save(out_path)
        if seq_id % 20 == 0:
            print(f"  saved: {out_path.name}")

    print(f"完了: {len(seq_ids)} シーケンスを {out_dir} に保存しました")


if __name__ == "__main__":
    main()
