import os
import glob
import shutil

os.makedirs('.\\dataset\\data', exist_ok=True)
age_list = os.listdir('.\\dataset\\10age')
for age in age_list:
    age_int = int(age)
    temp_age = (age_int // 10) * 10
    if temp_age >= 60:
        temp_age = 60
    os.makedirs(f'.\\dataset\\data\\{temp_age}', exist_ok=True)
    os.makedirs(f'.\\dataset\\data\\{temp_age}\\111', exist_ok=True)
    os.makedirs(f'.\\dataset\\data\\{temp_age}\\112', exist_ok=True)
    male_img_path = glob.glob(os.path.join('.\\dataset\\10age', f'{age}', '111', '*.jpg'))
    female_img_path = glob.glob(os.path.join('.\\dataset\\10age', f'{age}', '112', '*.jpg'))
    for path in male_img_path:
        shutil.move(path, os.path.join(f'.\\dataset\\data\\{temp_age}\\111', os.path.basename(path)))
    for path in female_img_path:
        shutil.move(path, os.path.join(f'.\\dataset\\data\\{temp_age}\\112', os.path.basename(path)))


