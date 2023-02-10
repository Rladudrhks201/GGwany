import os
import glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 원 폴더명 저장

def folder_name_save_and_name_change(file_path):
    labels = os.listdir(file_path)
    labelcode = []
    names = []

    for name in tqdm(labels):
        k = name.split('_')[0]
        v = name.split('_')[1]
        labelcode.append(k)
        names.append(v)
        os.rename(os.path.join(file_path, name), os.path.join(file_path, k))
    df = pd.DataFrame({'labelcode': labelcode, 'name': names})
    df.to_csv(f'.\\{os.path.basename(file_path)}_foldername.csv', index=False, encoding="utf-8-sig")

# 이미지 리사이즈
def image_resize(file_paths):
    for path in tqdm(file_paths, total=len(file_paths), desc='processing...'):
        img = Image.open(path)
        img = img.resize((640, 640))
        img.save(path)


if __name__=='__main__':
    # folder_name_save_and_name_change('D:\\dataset\\Training\\labels')
    # folder_name_save_and_name_change('D:\\dataset\\Training\\images')
    # folder_name_save_and_name_change('D:\\dataset\\Validation\\labels')
    # folder_name_save_and_name_change('D:\\dataset\\Validation\\images')

    # labelpths = glob.glob(os.path.join('D:\\New folder', '*'))
    # for path in labelpths:
    #     folder_name_save_and_name_change(path)

    # folder_name_save_and_name_change('D:\\kyk\\Training\\images')
    # folder_name_save_and_name_change('D:\\New folder\\[라벨]커피차1_train')


    # tr_ls = glob.glob(os.path.join('D:\\dataset', '*', 'images', '*', '*.jpg'))
    # image_resize(tr_ls)

    # tr_ls2 = glob.glob(os.path.join('D:\\kyk\\Training', 'images', '*', '*.jpg'))
    # image_resize(tr_ls2)

    d_img_path = glob.glob(os.path.join('D:\\dataset01', '*', 'images', '*.jpg'))
    # print(int(len(d_img_path) / 2))
    d_img_path = d_img_path[int(len(d_img_path) / 2) + 1:]
    image_resize(d_img_path)