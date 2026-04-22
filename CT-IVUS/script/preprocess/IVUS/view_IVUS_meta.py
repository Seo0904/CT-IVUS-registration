import pydicom
import numpy as np
from pathlib import Path

dicom_dir = Path("/workspace/data/org_data/IVUS/IVUS/DICOM")
base_save_dir = Path("/workspace/data/preprocessed/IVUS/check_meta")
base_save_dir.mkdir(parents=True, exist_ok=True)

# DICOMディレクトリ内の全ファイルを処理
dicom_files = sorted(dicom_dir.glob("*"))

for dicom_file in dicom_files:
    if dicom_file.is_file():
        # システムファイルや隠しファイルをスキップ
        if dicom_file.name.startswith('.'):
            print(f"Skipping: {dicom_file.name}")
            continue
        
        patient_id = dicom_file.name
        print(f"Processing: {patient_id}")
        
        # 患者ごとのディレクトリを作成
        patient_save_dir = base_save_dir / patient_id
        patient_save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            ds = pydicom.dcmread(dicom_file, force=True)
            
            # メタデータをテキストファイルに保存
            save_path = patient_save_dir / "metadata.txt"
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"=== DICOM Metadata for {patient_id} ===\n\n")
                f.write(str(ds))
                f.write("\n\n=== Pixel Array Info ===\n")
                try:
                    img = ds.pixel_array
                    f.write(f"Shape: {img.shape}\n")
                    f.write(f"Dtype: {img.dtype}\n")
                    f.write(f"Min: {np.min(img)}\n")
                    f.write(f"Max: {np.max(img)}\n")
                    f.write(f"Mean: {np.mean(img):.2f}\n")
                except Exception as e:
                    f.write(f"Error reading pixel array: {e}\n")
            
            print(f"  Saved metadata to: {save_path}")
            
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue

print(f"\nAll metadata saved to: {base_save_dir}")