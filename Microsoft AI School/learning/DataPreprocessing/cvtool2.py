import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = './images (2).jpeg'
# 이미지 읽기
image = cv2.imread(image_path)
# 색 반전을 막기 위해 RGB 타입 변환
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 사이즈 변환
image_50x50 = cv2.resize(image, (50, 50))
cv2.imwrite('./dog_image_50x50.jpeg', image_50x50)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image)
# ax[0].set_title('Original image')
# ax[1].imshow(image_50x50)
# ax[1].set_title('resized image')
# plt.show()

