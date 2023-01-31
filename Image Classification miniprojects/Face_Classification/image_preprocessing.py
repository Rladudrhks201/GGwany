from utils import *
import os
import glob
from PIL import Image
from PIL import ImageFile
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# file_path = glob.glob(os.path.join('.\\dataset\\data', '*', '*', '*.jpg'))

# Test
# img = Image.open('.\\dataset\\data\\20\\111\\9-0.jpg')
# img = expand2sqare(img, (0, 0, 0))
# img = img.resize((224, 224))
# img.show()

# Looking for Median of Image size

# image to square
# for path in file_path:
#     img = Image.open(path)
#     img = expand2sqare(img, (0, 0, 0))
#     img.save(path)

file_path = glob.glob(os.path.join('.\\dataset\\dataset', '*', '*', '*.jpg'))
size_list = []
for path in file_path:
    img = cv2.imread(path)
    size_list.append(img.shape[0])

print(np.median(size_list), np.mean(size_list))
# 87.0
# 데이터 추가 후 92, 평균은 134

