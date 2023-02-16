import os
import glob
import shutil
import random
from tqdm import tqdm

img_path = glob.glob(os.path.join('D:\\Large_label', '*', 'images', '*', '*.jpg'))
label_path = glob.glob(os.path.join('D:\\Large_label', '*', 'labels', '*', '*.txt'))
os.makedirs('.\\dataset\\test', exist_ok=True)


def split_data(label_no, p):
    os.makedirs("D:\\data001", exist_ok=True)
    label_path = glob.glob(os.path.join('D:\\Large_label', '*', 'labels', f'{label_no}'))
    for pth in label_path:
        dirlist = os.listdir(pth)
        tr_vl = pth.split('\\')[-3]
        os.makedirs(f'D:\\data001\\{tr_vl}', exist_ok=True)
        os.makedirs(f'D:\\data001\\{tr_vl}\\images', exist_ok=True)
        os.makedirs(f'D:\\data001\\{tr_vl}\\labels', exist_ok=True)
        random.seed(2324)
        after_path = random.sample(dirlist, int(p * len(dirlist)))
        for path1 in after_path:
            lstpth = glob.glob(os.path.join(pth, path1, '*.txt'))
            for path in lstpth:
                shutil.move(
                    os.path.join(f'D:\\dataset01\\{tr_vl}\\images', os.path.basename(path).replace('.txt', '.jpg')),
                    os.path.join(f'D:\\data001\\{tr_vl}\\images', os.path.basename(path).replace(".txt", ".jpg")))
                shutil.move(path,
                            os.path.join(f'D:\\data001\\{tr_vl}\\labels', f'{os.path.basename(path)}'))


def rest_file_move():
    nmb = [1, 2, 4, 6, 7, 8]
    for label_no in nmb:
        file_path = glob.glob(os.path.join('D:\\Large_label', '*', 'labels', f'{label_no}', '*', '*.txt'))
        for path in file_path:
            tr_vl = path.split('\\')[-5]
            shutil.move(
                os.path.join(f'D:\\dataset01\\{tr_vl}\\images', os.path.basename(path).replace('.txt', '.jpg')),
                os.path.join(f'D:\\data001\\{tr_vl}\\images', os.path.basename(path).replace(".txt", ".jpg")))
            shutil.move(path,
                        os.path.join(f'D:\\data001\\{tr_vl}\\labels', f'{os.path.basename(path)}'))




if __name__ == '__main__':
    split_data(label_no=0, p=0.33)
    split_data(label_no=3, p=0.5)
    split_data(label_no=5, p=0.5)
    rest_file_move()