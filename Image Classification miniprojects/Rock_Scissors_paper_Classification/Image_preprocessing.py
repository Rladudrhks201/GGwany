import matplotlib.pyplot as plt
from utils import *
import os
import glob
from PIL import Image

# Test
# img = Image.open('.\\dataset\\train\\paper\\0cb6cVL8pkfi4wF6.png').convert('RGB')
# img = expand2sqare(img, (0, 0, 0))
# plt.imshow(img)
# plt.show()

file_path = glob.glob(os.path.join('.\\dataset', '*', '*', '*.png'))
for path in file_path:
    img = Image.open(path)
    img = expand2sqare(img, (0, 0, 0))
    img.save(path)