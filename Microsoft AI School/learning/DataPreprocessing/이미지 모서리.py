# 모서리 감지
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './edge.png'
image_read = cv2.imread(image_path)

image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

block_size = 2  # 모서리 감지 매개 변수 설정
aperture = 29
free_parameter = 0.04

detector_response = cv2.cornerHarris(
    image_gray, block_size, aperture, free_parameter
)  # 모서리를 감지
print(detector_response)
# 임계값보다 큰 감지 결과만 남기고 흰색으로 표시합니다
threhold = 0.02
image_read[detector_response > threhold * detector_response.max()] = [255, 255, 255]

image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
image_show(image_gray)
