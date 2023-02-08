import cv2
import os
import glob
import shutil
from tqdm import tqdm

def file_move(path, errorpth):
    full_paths = glob.glob(os.path.join(path, '*', '*', '*', '*'))
    count = 0
    for path1 in full_paths:
        file = os.path.basename(path1)
        folder = path1.split('\\')[-4]
        dtype = path1.split('\\')[-3]
        type = path1.split('\\')[-2]
        error_path = f"{errorpth}\\{folder}\\{dtype}\\{type}\\{file}"
        if os.path.exists(f"{path}\\{folder}\\{dtype}\\{file}"):

            print(f"{path}\\{folder}\\{dtype}\\{file}")
            os.makedirs(f"{errorpth}\\{folder}", exist_ok=True)
            os.makedirs(f"{errorpth}\\{folder}\\{dtype}", exist_ok=True)
            os.makedirs(f"{errorpth}\\{folder}\\{dtype}\\{type}", exist_ok=True)
            shutil.move(path1, error_path)
        else:
            shutil.move(path1, f"{path}\\{folder}\\{dtype}\\{file}")
            count += 1




if __name__=='__main__':
    path = 'D:\\dataset01'
    error_path = 'D:\\errors'


    file_move(path, error_path)