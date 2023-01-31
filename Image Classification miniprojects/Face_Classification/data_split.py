import glob
import os
from sklearn.model_selection import train_test_split
import cv2

os.makedirs('.\\dataset\\dataset', exist_ok=True)
os.makedirs('.\\dataset\\dataset\\train', exist_ok=True)
os.makedirs('.\\dataset\\dataset\\test', exist_ok=True)
os.makedirs('.\\dataset\\dataset\\val', exist_ok=True)

def split_data(age, gender, file_path):
    y_temp = []
    data_path = glob.glob(os.path.join(file_path, f'{age}', f'{gender}', '*.jpg'))
    for path in data_path:
        y_age = path.split('\\')[-3]
        y_gender = path.split('\\')[-2]
        y_gender = int(y_gender) - 111
        y_temp.append(f'{y_age}_{y_gender}')

    x_tr_temp, x_val_temp, y_tr_temp, y_val_temp = train_test_split(data_path, y_temp, train_size=0.9,
                                                                  random_state=2324)

    for path, label in zip(x_tr_temp, y_tr_temp):
        os.makedirs(f'.\\dataset\\dataset\\train\\{label}', exist_ok=True)
        img = cv2.imread(path)
        cv2.imwrite(f'.\\dataset\\dataset\\train\\{label}\\{os.path.basename(path).split(".")[0]}.jpg', img)

    for path, label in zip(x_val_temp, y_val_temp):
        os.makedirs(f'.\\dataset\\dataset\\val\\{label}', exist_ok=True)
        img = cv2.imread(path)
        cv2.imwrite(f'.\\dataset\\dataset\\val\\{label}\\{os.path.basename(path).split(".")[0]}.jpg', img)




age_list = os.listdir('.\\dataset\\data')
gender_list = ['111', '112']
# print(age_list)

for age in age_list:
    for sex in gender_list:
        split_data(age, sex, '.\\dataset\\data')

