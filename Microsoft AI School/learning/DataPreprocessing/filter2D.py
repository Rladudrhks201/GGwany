# 이미지 블러 처리
# filter2D 메서드 사용
import cv2
from utils import image_show
import numpy as np

# 이미지 경로
image_path = './images.jpeg'

# 이미지 읽기 처리
image = cv2.imread(image_path)
# print(image)

# 커널 생성 처리
kernel = np.ones((5, 5)) / 25.0  # 모두 더하면 1이 되도록 정규화
image_kernel = cv2.filter2D(image, -1, kernel)
image_show(image_kernel)
