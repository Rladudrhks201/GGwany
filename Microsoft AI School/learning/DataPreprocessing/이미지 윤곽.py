# canny
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './images.jpeg'
image = cv2.imread(image_path)

# 경계선 찾기
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 픽셀 강도의 중간값을 계산
mdeian_intensity = np.median(image_gray)

# 중간 픽셀 강도에서 위아래 1표준편차 떨어진 값을 임계값으로 설정
lower_threshold = int(max(0, (1.0 - 0.1) * mdeian_intensity))
upper_threshold = int(min(0, (1.0 + 0.1) * mdeian_intensity))
# 0.33을 여러 숫자를 넣어서 실험해야함 0.33

#canny edge detection 적용
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
image_show(image_canny)
