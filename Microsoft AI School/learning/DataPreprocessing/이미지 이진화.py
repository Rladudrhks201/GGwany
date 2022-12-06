# 이미지 이진화
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './images.jpeg'

# 이미지 이진화
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
max_output_value = 255
neighborhood_size = 133 # 홀수만 지정, 99
subtract_from_mean = 10

image_binary = cv2.adaptiveThreshold(image_gray, max_output_value,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, neighborhood_size,
                                     subtract_from_mean)
image_show(image_binary)