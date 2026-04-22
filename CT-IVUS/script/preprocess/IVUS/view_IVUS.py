import pydicom
import numpy as np
from PIL import Image
from pathlib import Path

dicom_dir = Path("/workspace/data/org_data/IVUS/IVUS/DICOM")
# dicom_dir = Path("/workspace/data/org_data/IVUS/IVUS")
base_save_dir = Path("/workspace/data/preprocessed/IVUS/check_png")

# DICOMディレクトリ内の全ファイルを処理
dicom_files = sorted(dicom_dir.glob("*"))

for dicom_file in dicom_files:
    if dicom_file.is_file():
        # システムファイルや隠しファイルをスキップ
        if dicom_file.name.startswith('.'):
            print(f"Skipping: {dicom_file.name}")
            continue
        
        # 患者番号ごとのディレクトリを作成
        patient_id = dicom_file.name
        save_dir = base_save_dir / patient_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {patient_id}")
        
        try:
            ds = pydicom.dcmread(dicom_file, force=True)
            img = ds.pixel_array  # ここで (H,W) or (T,H,W) が返ることが多い
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue
        print(f"  shape: {img.shape}, dtype: {img.dtype}, min: {np.min(img)}, max: {np.max(img)}")
        
        if img.ndim == 2:
            # 2D画像の場合
            save_path = save_dir / f"{patient_id}.png"
            Image.fromarray(img).save(save_path)
            print(f"  Saved to: {save_dir}")
        else:
            # 複数フレームの場合は全フレームを保存
            for i in range(img.shape[0]):
                save_path = save_dir / f"frame{i:05d}.png"
                Image.fromarray(img[i]).save(save_path)
            print(f"  Saved {img.shape[0]} frames to: {save_dir}")

print(f"\nAll images saved to: {base_save_dir}")
