# 이미지를 선명하게 변경
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './images.jpeg'

# 이미지를 선명하게 ( 커널 생성 : 대상이 있는 픽셀 강조)
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

# RGB 타입으로 변경 , plt 활용을 위해
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 커널 생성
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# 커널 적용
image_sharp = cv2.filter2D(image_rgb, -1, kernel)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_rgb)
ax[0].set_title('original')
ax[1].imshow(image_sharp)
ax[1].set_title('sharp')
plt.show()
