# 이미지 대비 높이기
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './images.jpeg'
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(image_gray)

# 회색 대비 plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_gray, cmap='gray')
ax[0].set_title('original')
ax[1].imshow(image_enhanced, cmap='gray')
ax[1].set_title('enhanced')
plt.show()

# 컬러 이미지 대비 높이기 : rgb -> yuv -> equalizehist() -> rgb
image = cv2.imread(image_path)  # BGR
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)  # YUV

# 히스토그램 평활화
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# RGB 변경 , plt 적용을 위해
image_rgb_temp = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_rgb)
ax[0].set_title('original')
ax[1].imshow(image_rgb_temp)
ax[1].set_title('enhanced')
plt.show()
