import glob
import os
from sklearn.model_selection import train_test_split
import cv2
import shutil


def split_data(file_path):
    os.makedirs('D:\\Test', exist_ok=True)
    os.makedirs('D:\\Test\\images', exist_ok=True)
    os.makedirs('D:\\Test\\labels', exist_ok=True)
    image_path = glob.glob(os.path.join(file_path, 'images', '*.jpg'))
    label_path = glob.glob(os.path.join(file_path, 'labels', '*.json'))

    x_val_temp, x_te_temp, y_val_temp, y_te_temp = train_test_split(image_path, label_path, train_size=0.5,
                                                                    random_state=7727)

    for img_path, label_path in zip(x_te_temp, y_te_temp):
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        shutil.move(img_path, f'D:\\Test\\images\\{img_name}')
        shutil.move(label_path, f'D:\\Test\\labels\\{label_name}')


if __name__ == '__main__':
    split_data('D:\\Valid')
