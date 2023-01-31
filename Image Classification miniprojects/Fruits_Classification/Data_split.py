from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import shutil

# 데이터 나누기
# Train 80, Validation 10, Test 10

"""
data
    - train
        -dekopon
        -kanpei
        -kiwi
        -orange
    - val
        -dekopon
        -kanpei
        -kiwi
        -orange
    - test
        -dekopon
        -kanpei
        -kiwi
        -orange
    
"""

base_img_path = 'C:\\Users\\user\\Desktop\\Search\\Data'
train_img_path = os.path.join(base_img_path, 'train')
val_img_path = os.path.join(base_img_path, 'val')
test_img_path = os.path.join(base_img_path, 'test')

dekopon_data = glob.glob(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\dekopon\\resize', '*.png'))
kanpei_data = glob.glob(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\kanpei\\resize', '*.png'))
orange_data = glob.glob(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\orange\\resize', '*.png'))
kiwi_data = glob.glob(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\kiwi\\resize', '*.png'))

# 데이터의 수가 많지 않아 Test data는 따로 나누지 않음
dekopon_train_data, dekopon_val_data = train_test_split(dekopon_data, test_size=0.1, random_state=6262)
kanpei_train_data, kanpei_val_data = train_test_split(kanpei_data, test_size=0.1, random_state=6262)
kiwi_train_data, kiwi_val_data = train_test_split(kiwi_data, test_size=0.1, random_state=6262)
orange_train_data, orange_val_data = train_test_split(orange_data, test_size=0.1, random_state=6262)

# 경로 생성
path_list = [train_img_path, val_img_path]
fruit_list = ['dekopon', 'kanpei', 'kiwi', 'orange']
for i in path_list:
    os.makedirs(i, exist_ok=True)
    for j in fruit_list:
        image_path = os.path.join(i, j)
        os.makedirs(image_path, exist_ok=True)

# 이미지 저장
for dekopon_train_path in dekopon_train_data:
    file_name1 = os.path.basename(dekopon_train_path)
    shutil.move(dekopon_train_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\train\\dekopon\\{file_name1}')

for dekopon_val_path in dekopon_val_data:
    file_name2 = os.path.basename(dekopon_val_path)
    shutil.move(dekopon_val_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\val\\dekopon\\{file_name2}')

for kanpei_train_path in kanpei_train_data:
    file_name1 = os.path.basename(kanpei_train_path)
    shutil.move(kanpei_train_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\train\\kanpei\\{file_name1}')

for kanpei_val_path in kanpei_val_data:
    file_name2 = os.path.basename(kanpei_val_path)
    shutil.move(kanpei_val_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\val\\kanpei\\{file_name2}')

for orange_train_path in orange_train_data:
    file_name1 = os.path.basename(orange_train_path)
    shutil.move(orange_train_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\train\\orange\\{file_name1}')

for orange_val_path in orange_val_data:
    file_name2 = os.path.basename(orange_val_path)
    shutil.move(orange_val_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\val\\orange\\{file_name2}')

for kiwi_train_path in kiwi_train_data:
    file_name1 = os.path.basename(kiwi_train_path)
    shutil.move(kiwi_train_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\train\\kiwi\\{file_name1}')

for kiwi_val_path in kiwi_val_data:
    file_name2 = os.path.basename(kiwi_val_path)
    shutil.move(kiwi_val_path, f'C:\\Users\\user\\Desktop\\Search\\Data\\val\\kiwi\\{file_name2}')


