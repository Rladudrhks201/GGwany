import os
import glob
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

val_path = glob.glob(os.path.join('.\\dataset\\valid', '*', '*.jpg'))
os.makedirs('.\\dataset\\test', exist_ok=True)
def split_test_data(file_path='.\\dataset\\valid'):
    y_temp = []
    data_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
    for path in data_path:
        y_data = path.split('\\')[-2]
        y_temp.append(y_data)

    x_val_temp, x_te_temp, y_val_temp, y_te_temp = train_test_split(data_path, y_temp, train_size=0.5,
                                                                    random_state=2324)
    for path, label in zip(x_te_temp, y_te_temp):
        os.makedirs(f'.\\dataset\\test\\{label}', exist_ok=True)
        shutil.move(path,
                    os.path.join(f'.\\dataset\\test\\{label}', f'{os.path.basename(path).split(".")[0]}.jpg'))

if __name__ == '__main__':
    split_test_data()