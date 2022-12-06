import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.png')

# creating maxican hat filter for
# 5x5
filter = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0],
                   [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0],
                   [0, 0, -1, 0, 0]])

# 3x3 , 필터 크기에 따라 내용이 달라짐, 파일로 다운받아놓음
# filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

mexican_hat_image = cv2.filter2D(image, -1, filter)
image_show(mexican_hat_image)
