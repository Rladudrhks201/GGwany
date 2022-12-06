import cv2
import numpy as np
from utils import image_show

img_path = "./car.png"
img = cv2.imread(img_path)

# 엠보싱효과
filter1 = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
filter2 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
emboss_img = cv2.filter2D(img, -1, filter1)
emboss_img = emboss_img + 128 # 결과가 너무 검게 나오지 않게 회색으로

image_show(emboss_img)
