import cv2
import numpy as np
import os
from PIL import Image


def expand2sqare(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def image_file(image_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(image_folder_path):
        for filename in files:
            # image.jpg -> .jpg
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                root = os.path.join(path, filename)

                all_root.append(root)
            else:
                print('no image files')
                continue
    return all_root


folder = ['apple_high_res', 'kiwi_high_res', 'pineapple_high_res']
for i in folder:
    image_path_list = image_file(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\', i))

    for image_path in image_path_list:
        # print(image_path)
        # image_name_temp = image_path.split('\\')
        # print(image_name_temp[-1])
        image_name_temp = os.path.basename(image_path)
        image_name_temp = image_name_temp.replace(".jpg", "")
        if '사과' in image_name_temp:
            image_name_temp.replace('사과', 'apple')
        elif '키위' in image_name_temp:
            image_name_temp.replace('키위', 'kiwi')
        elif '파인애플' in image_name_temp:
            image_name_temp.replace('파인애플', 'pineapple')
        # print(image_name_temp)
        # image_name_temp2 = os.path.abspath(image_path) # 절대경로

        img = Image.open(image_path)
        img_new = expand2sqare(img, (0, 0, 0)).resize((224, 224))
        os.makedirs(os.path.join(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\', i), 'resize'), exist_ok=True)
        save_path = os.path.join(os.path.join('C:\\Users\\user\\Desktop\\Search\\Data\\', i),
                                 f"resize\\{image_name_temp}.png")
        img_new.save(save_path, quality=100)
        # 경로 변경하면 끝
