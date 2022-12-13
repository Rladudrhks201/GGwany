# image data get -> train 80 val 10 test 10

# image -> cat, dog get

import os
import glob
import cv2
import natsort
from sklearn.model_selection import train_test_split
# pip install natsort
cat_image_path = 'C:\\Users\\user\\Desktop\\project\\image_cat_dog\\cats'
dog_image_path = 'C:\\Users\\user\\Desktop\\project\\image_cat_dog\\dogs'

cat_image_full_path = natsort.natsorted(    # 무작위가 아닌 정렬하여 데이터를 불러오기
    glob.glob(os.path.join(f'{cat_image_path}/*.jpg'))  # 경로에 있는 모든 파일 불러오기
)
# print(len(cat_image_full_path))

dog_image_full_path = natsort.natsorted(    # 무작위가 아닌 정렬하여 데이터를 불러오기
    glob.glob(os.path.join(f'{dog_image_path}/*.jpg'))  # 경로에 있는 모든 파일 불러오기
)
# print(dog_image_full_path)

# train 80 val 20 -> val 10 test 10
cat_train_data, cat_val_data = train_test_split(
    cat_image_full_path, test_size=0.2, random_state=7777)
cat_val, cat_test = train_test_split(
    cat_val_data, test_size=0.5, random_state=7777)

print(
    f"cat train data : {len(cat_train_data)}, cat val data : {len(cat_val)}, cat test data : {len(cat_test)}")
# cat train data : 3200, cat val data : 400, cat test data : 400

# train 80 val 20 -> val 10 test 10
dog_train_data, dog_val_data = train_test_split(
    dog_image_full_path, test_size=0.2, random_state=7777)
dog_val, dog_test = train_test_split(
    dog_val_data, test_size=0.5, random_state=7777)

print(
    f"dog train data : {len(dog_train_data)}, dog val data : {len(dog_val)}, dog test data : {len(dog_test)}")
# dog train data : 3200, dog val data : 400, dog test data : 400

# for cat_train_data_path in cat_train_data:
#     img = cv2.imread(cat_train_data_path)
#     os.makedirs('C:\\Users\\user\\Desktop\\project\\dataset\\train\\cat\\', exist_ok=True)
#     file_name = os.path.basename(cat_train_data_path)
#     cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\dataset\\train\\cat\\{file_name}', img)
#
# for cat_val_path, cat_test_path in zip(cat_val, cat_test):
#     img_val = cv2.imread(cat_val_path)
#     img_test = cv2.imread(cat_test_path)
#     file_name_val = os.path.basename(cat_val_path)
#     file_name_test = os.path.basename(cat_test_path)
#     os.makedirs('C:\\Users\\user\\Desktop\\project\\dataset\\val\\cat\\', exist_ok=True)
#     cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\dataset\\val\\cat\\{file_name_val}', img_val)
#     os.makedirs('C:\\Users\\user\\Desktop\\project\\dataset\\test\\cat\\', exist_ok=True)
#     cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\dataset\\test\\cat\\{file_name_test}', img_test)


for dog_train_data_path in dog_train_data:
    img = cv2.imread(dog_train_data_path)
    os.makedirs('C:\\Users\\user\\Desktop\\project\\dataset\\train\\dog\\', exist_ok=True)
    file_name = os.path.basename(dog_train_data_path)
    cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\dataset\\train\\dog\\{file_name}', img)

for dog_val_path, dog_test_path in zip(dog_val, dog_test):
    img_val = cv2.imread(dog_val_path)
    img_test = cv2.imread(dog_test_path)
    file_name_val = os.path.basename(dog_val_path)
    file_name_test = os.path.basename(dog_test_path)
    os.makedirs('C:\\Users\\user\\Desktop\\project\\dataset\\val\\dog\\', exist_ok=True)
    cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\dataset\\val\\dog\\{file_name_val}', img_val)
    os.makedirs('C:\\Users\\user\\Desktop\\project\\dataset\\test\\dog\\', exist_ok=True)
    cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\dataset\\test\\dog\\{file_name_test}', img_test)