# 이미지 회전
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './images.jpeg'
image = cv2.imread(image_path)

# 회전
img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 시계방향 90
img180 = cv2.rotate(image, cv2.ROTATE_180) # 180
img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE) # 270

cv2.imshow("original", image)
cv2.imshow("90 degree", img90)
cv2.imshow("180 degree", img180)
cv2.imshow("270 degree", img270)
cv2.waitKey(0)