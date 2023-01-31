import glob
import os
from PIL import Image
import shutil
from utils import *

image_path = glob.glob(os.path.join('.\\dataset\\new data', '*', '*.jpg'))
def image_folder(image_path):
    gender_list = os.listdir(image_path)
    for gender in gender_list:
        image_path = glob.glob(os.path.join('.\\dataset\\new data', f'{gender}', '*.jpg'))
        for path in image_path:
            name = os.path.basename(path)
            age_temp = name.split('.jpg')[0].split('A')[1]
            if int(age_temp) < 40:
                continue
            elif int(age_temp) >= 60:
                age_temp = 60
            age_label = (int(age_temp) // 10) * 10
            img = Image.open(path)
            img = expand2sqare(img, (0, 0, 0))
            img.save(os.path.join(f'.\\dataset\\data\\{age_label}\\{gender}',
                                           os.path.basename(path).replace('A', 'B')))


image_folder('.\\dataset\\new data')