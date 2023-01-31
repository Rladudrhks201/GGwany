import os
import glob
import shutil
import pandas as pd

# 이미지 소분류 대분류로 합치기

train_image_path = glob.glob(os.path.join('D:\\Data\\product_image\\Training\\image', '*', '*', '*.jpg'))
val_image_path = glob.glob(os.path.join('D:\\Data\\product_image\\Validation\\image', '*', '*', '*.jpg'))


for path in train_image_path:
    file_name1 = os.path.basename(path)
    shutil.move(path, os.path.join('D:\\Data\\product_image\\Training\\image', path.split('\\')[-3], f'{file_name1}'))

for path in val_image_path:
    file_name1 = os.path.basename(path)
    shutil.move(path, os.path.join('D:\\Data\\product_image\\Validation\\image', path.split('\\')[-3], f'{file_name1}'))



# 한글명 폴더 영어폴더로 변경
df_train = pd.read_csv('.\\folder name.csv')
df_val = pd.read_csv('.\\folder name2.csv')

for dir in os.listdir('D:\\Data\\product_image\\Training\\image'):
    if '원천' in dir:
        dir_name = dir.split(']')[1]

        path = df_train[df_train['원 폴더명'] == dir_name]['바뀐 폴더명'].values
        path = str(path).lstrip('[').rstrip(']').strip("'")
        print(path)
        new_path = os.path.join('D:\\Data\\product_image\\Training\\image', f'{str(path)}')
        print(new_path)
        os.rename(os.path.join('D:\\Data\\product_image\\Training\\image', dir), new_path)

for dir in os.listdir('D:\\Data\\product_image\\Validation\\image'):
    if '원천' in dir:
        dir_name = dir.split(']')[1]

        path = df_val[df_val['원 폴더명'] == dir_name]['바뀐 폴더명'].values
        path = str(path).lstrip('[').rstrip(']').strip("'")
        new_path = os.path.join('D:\\Data\\product_image\\Validation\\image', f'{str(path)}')
        os.rename(os.path.join('D:\\Data\\product_image\\Validation\\image', dir), new_path)

train_image_path = glob.glob(os.path.join('D:\\Data\\product_image\\Training\\image', '*', '*.jpg'))
val_image_path = glob.glob(os.path.join('D:\\Data\\product_image\\Validation\\image', '*', '*.jpg'))
print(len(train_image_path), len(val_image_path))