import pandas as pd
import os

# Excelファイルのパス
excel_path = '/workspace/data/org_data/table/編集用データkVp編集20251111.xlsx'
sheet_name = '画像あり(追加後)'

# 出力ディレクトリ
output_dir = '/workspace/data/preprocessed/IVUS_csv'
os.makedirs(output_dir, exist_ok=True)

# Excelファイルを読み込む
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# 必要な列を確認（存在しない場合はエラー）
required_columns = ['ID', 'futsuka_jokyo', 'tiryo_bui_1', 'tiryo_bui_2', 'tiryo_bui_3']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the sheet.")

# futsuka_jokyo に値がある行
df_present = df[df['futsuka_jokyo'].notna() & (df['futsuka_jokyo'] != '')][['ID', 'futsuka_jokyo']]

# futsuka_jokyo に値がない行
df_absent = df[df['futsuka_jokyo'].isna() | (df['futsuka_jokyo'] == '')][['ID', 'tiryo_bui_1', 'tiryo_bui_2', 'tiryo_bui_3']]

# CSVに保存
present_csv_path = os.path.join(output_dir, 'futsuka_jokyo_present.csv')
absent_csv_path = os.path.join(output_dir, 'futsuka_jokyo_absent.csv')

df_present.to_csv(present_csv_path, index=False, encoding='utf-8-sig')
df_absent.to_csv(absent_csv_path, index=False, encoding='utf-8-sig')

print(f"Processed data saved to {present_csv_path} and {absent_csv_path}")
