import cv2
import os
import glob
import argparse
from PIL import Image

# image pretreatment
image_path = 'C:\\Users\\user\\Desktop\\Search\\Data'


# orange 오렌지
# kanpei 레드향
# dekopon 한라봉
# kiwi 키위

def image_file_check(opt):
    image_path = opt.image_folder_path
    # 오렌지
    orange_file_path = glob.glob(os.path.join(image_path, 'orange', '*.jpg'))
    # 레드향
    kanpai_file_path = glob.glob(os.path.join(image_path, 'kanpei', '*.jpg'))
    # 한라봉
    dekopon_file_path = glob.glob(os.path.join(image_path, 'dekopon', '*.jpg'))
    # 키위
    kiwi_file_path = glob.glob(os.path.join(image_path, 'kiwi', '*.jpg'))

    print('오렌지 파일 개수', len(orange_file_path))
    print('레드향 파일 개수', len(kanpai_file_path))
    print('한라봉 파일 개수', len(dekopon_file_path))
    print('키위 파일 개수', len(kiwi_file_path))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder-path", type=str, default='C:\\Users\\user\\Desktop\\Search\\Data')
    opt = parser.parse_args()
    return opt


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


def image_resize(opt):
    image_path = opt.image_folder_path
    # 오렌지
    orange_file_path = glob.glob(os.path.join(image_path, 'orange', '*.jpg'))
    # 레드향
    kanpai_file_path = glob.glob(os.path.join(image_path, 'kanpei', '*.jpg'))
    # 한라봉
    dekopon_file_path = glob.glob(os.path.join(image_path, 'dekopon', '*.jpg'))
    # 키위
    kiwi_file_path = glob.glob(os.path.join(image_path, 'kiwi', '*.jpg'))
    for image_paths in orange_file_path:
        image_name_temp = os.path.basename(image_paths).replace('.jpg', '')
        img = Image.open(image_paths)
        img_new = expand2sqare(img, (0, 0, 0)).resize((400, 400))
        temp_path = 'C:\\Users\\user\\Desktop\\Search\\Data\\orange'
        os.makedirs(os.path.join(temp_path, 'resize'), exist_ok=True)
        save_path = os.path.join(temp_path, 'resize', f"{image_name_temp}.png")
        img_new.save(save_path, quality=100)
    for image_paths in dekopon_file_path:
        image_name_temp = os.path.basename(image_paths).replace('.jpg', '')
        img = Image.open(image_paths)
        img_new = expand2sqare(img, (0, 0, 0)).resize((400, 400))
        temp_path = 'C:\\Users\\user\\Desktop\\Search\\Data\\dekopon'
        os.makedirs(os.path.join(temp_path, 'resize'), exist_ok=True)
        save_path = os.path.join(temp_path, 'resize', f"{image_name_temp}.png")
        img_new.save(save_path, quality=100)
    for image_paths in kanpai_file_path:
        image_name_temp = os.path.basename(image_paths).replace('.jpg', '')
        img = Image.open(image_paths)
        img_new = expand2sqare(img, (0, 0, 0)).resize((400, 400))
        temp_path = 'C:\\Users\\user\\Desktop\\Search\\Data\\kanpei'
        os.makedirs(os.path.join(temp_path, 'resize'), exist_ok=True)
        save_path = os.path.join(temp_path, 'resize', f"{image_name_temp}.png")
        img_new.save(save_path, quality=100)
    for image_paths in kiwi_file_path:
        image_name_temp = os.path.basename(image_paths).replace('.jpg', '')
        img = Image.open(image_paths)
        img_new = expand2sqare(img, (0, 0, 0)).resize((400, 400))
        temp_path = 'C:\\Users\\user\\Desktop\\Search\\Data\\kiwi'
        os.makedirs(os.path.join(temp_path, 'resize'), exist_ok=True)
        save_path = os.path.join(temp_path, 'resize', f"{image_name_temp}.png")
        img_new.save(save_path, quality=100)


if __name__ == '__main__':
    opt = parse_opt()
    image_file_check(opt)
    image_resize(opt)
