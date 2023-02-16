import os
import glob
import pandas as pd

label_csv = pd.read_csv('.\\small_category.csv', encoding='utf-8-sig')
label_csv = label_csv[label_csv['div_l'] == '주류']
df = label_csv[['code', 'labels']]

for i in range(len(df)):
    label = df.iloc[i, :]["code"]
    encoded_label = df.iloc[i, :]["labels"]
    with open('.\\drink_label.txt', 'a') as f:
        f.write(f'{encoded_label}: {label}\n')
