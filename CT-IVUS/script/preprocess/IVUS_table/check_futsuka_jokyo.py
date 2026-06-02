import pandas as pd
import os

# CSVファイルのパス
csv_path = '/workspace/data/org_data/table/ccti_calc_data0427.csv'

# 出力ディレクトリ
output_dir = '/workspace/data/preprocessed/IVUS_csv'
os.makedirs(output_dir, exist_ok=True)

# CSVファイルを読み込む
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 必要な列を確認（存在しない場合はエラー）
required_columns = ['gazou_ID', 'futsuka_new', 'futsuka_jokyo']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the CSV.")

# futsuka_new が 1 の行を抽出し、gazou_ID と futsuka_jokyo を取得
df_futsuka = df[df['futsuka_new'] == 1][['gazou_ID', 'futsuka_jokyo']]

# CSVに保存
output_path = os.path.join(output_dir, 'futsuka_jokyo.csv')
df_futsuka.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"抽出件数: {len(df_futsuka)} 件")
print(f"保存先: {output_path}")
