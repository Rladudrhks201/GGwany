import os
import glob

from sklearn.model_selection import train_test_split
import cv2

# 데이터 분리
# 학습 중간평가 테스트
# 학습 8 중간평가 1 테스트 1
# 데이터 구조
"""
dataset
    - train
        - 각 라벨별 폴더
    
    - val
        - 각 라벨별 폴더
    
    - test
        - 각 라벨별 폴더
"""
os.makedirs('.\\dataset', exist_ok=True)
os.makedirs('.\\dataset\\train', exist_ok=True)
os.makedirs('.\\dataset\\test', exist_ok=True)
os.makedirs('.\\dataset\\val', exist_ok=True)
y_cloudy = []
y_desert = []
y_green_area = []
y_water = []

cloudy_data_path = glob.glob(os.path.join('.\\', 'data', 'cloudy', '*.jpg'))
desert_data_path = glob.glob(os.path.join('.\\', 'data', 'desert', '*.jpg'))
green_area_data_path = glob.glob(os.path.join('.\\', 'data', 'green_area', '*.jpg'))
water_data_path = glob.glob(os.path.join('.\\', 'data', 'water', '*.jpg'))

for path in cloudy_data_path:
    y_data = path.split('\\')[-2]
    y_cloudy.append(y_data)

for path in desert_data_path:
    y_data = path.split('\\')[-2]
    y_desert.append(y_data)

for path in green_area_data_path:
    y_data = path.split('\\')[-2]
    y_green_area.append(y_data)

for path in water_data_path:
    y_data = path.split('\\')[-2]
    y_water.append(y_data)

x_tr_cloudy, x_tp_cloudy, y_tr_cloudy, y_tp_cloudy = train_test_split(cloudy_data_path, y_cloudy, train_size=0.8,
                                                                      random_state=2324)
x_val_cloudy, x_te_cloudy, y_val_cloudy, y_te_cloudy = train_test_split(x_tp_cloudy, y_tp_cloudy, train_size=0.5,
                                                                        random_state=2324)

x_tr_desert, x_tp_desert, y_tr_desert, y_tp_desert = train_test_split(desert_data_path, y_desert, train_size=0.8,
                                                                      random_state=2324)
x_val_desert, x_te_desert, y_val_desert, y_te_desert = train_test_split(x_tp_desert, y_tp_desert, train_size=0.5,
                                                                        random_state=2324)

x_tr_green_area, x_tp_green_area, y_tr_green_area, y_tp_green_area = train_test_split(green_area_data_path,
                                                                                      y_green_area, train_size=0.8,
                                                                                      random_state=2324)
x_val_green_area, x_te_green_area, y_val_green_area, y_te_green_area = train_test_split(x_tp_green_area,
                                                                                        y_tp_green_area, train_size=0.5,
                                                                                        random_state=2324)

x_tr_water, x_tp_water, y_tr_water, y_tp_water = train_test_split(water_data_path, y_water, train_size=0.8,
                                                                  random_state=2324)
x_val_water, x_te_water, y_val_water, y_te_water = train_test_split(x_tp_water, y_tp_water, train_size=0.5,
                                                                    random_state=2324)
# print(len(x_tr_water), len(x_tr_cloudy))
# print(len(x_te_water), len(x_te_cloudy))

for path, label in zip(x_tr_cloudy, y_tr_cloudy):
    os.makedirs(f'.\\dataset\\train\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\train\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_val_cloudy, y_val_cloudy):
    os.makedirs(f'.\\dataset\\val\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\val\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_te_cloudy, y_te_cloudy):
    os.makedirs(f'.\\dataset\\test\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\test\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_tr_desert, y_tr_desert):
    os.makedirs(f'.\\dataset\\train\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\train\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_val_desert, y_val_desert):
    os.makedirs(f'.\\dataset\\val\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\val\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_te_desert, y_te_desert):
    os.makedirs(f'.\\dataset\\test\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\test\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_tr_green_area, y_tr_green_area):
    os.makedirs(f'.\\dataset\\train\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\train\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_val_green_area, y_val_green_area):
    os.makedirs(f'.\\dataset\\val\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\val\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_te_green_area, y_te_green_area):
    os.makedirs(f'.\\dataset\\test\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\test\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_tr_water, y_tr_water):
    os.makedirs(f'.\\dataset\\train\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\train\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_val_water, y_val_water):
    os.makedirs(f'.\\dataset\\val\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\val\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

for path, label in zip(x_te_water, y_te_water):
    os.makedirs(f'.\\dataset\\test\\{label}', exist_ok=True)
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\test\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

