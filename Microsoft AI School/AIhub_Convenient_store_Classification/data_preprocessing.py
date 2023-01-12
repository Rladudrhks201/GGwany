from utils import *
import os
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from multiprocessing import Pool

file_path1 = glob.glob(os.path.join('D:\\Data\\product_image', '*', 'image', '*', '*.jpg'))
os.makedirs('.\\dataset', exist_ok=True)



def image_preprocess(file_path):
    for path in file_path:
        img = Image.open(path)
        img = expand2sqare(img, (0, 0, 0))
        img = img.resize((224, 224))
        img.save(path)

image_preprocess(file_path1)


