# 이미지 좌우 반전
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './images.jpeg'
image = cv2.imread(image_path)

# 반전
image_lr = cv2.flip(image, 1)  # 1은 좌우 0은 상하
image_show(image_lr)
image_ud = cv2.flip(image, 0)
image_show(image_ud)
