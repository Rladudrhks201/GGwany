# 기본적인 이미지 처리 기술을 이용한 이미지 선명화 -3
import cv2
import numpy as np
from utils import image_show

img = cv2.imread('./car.png', 0)

# Creating out sharpening filter
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sharpen_img = cv2.filter2D(img, -1, filter)
# -1 은 입력 이미지 크기와 동일하게 나옴

cv2.imshow("original", img)
cv2.waitKey(0)
image_show(sharpen_img)



