# 가우시안 블러
import numpy as np
from utils import image_show
import cv2

# 이미지 경로 및 읽기
image_path = './images.jpeg'
image = cv2.imread(image_path)

image_g_blur = cv2.GaussianBlur(image, (5, 5), 0)
# 블러 튜플의 값은 홀수만 가능
image_show(image_g_blur)
