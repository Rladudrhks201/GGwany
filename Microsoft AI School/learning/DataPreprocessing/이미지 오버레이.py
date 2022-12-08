# 같은 크기의 이비지 블렌딩
import cv2
import matplotlib.pyplot as plt
import numpy as np

large_img = cv2.imread('./tennis.png')
watermark = cv2.imread('./goat.png')

print('large image', large_img.shape)
print('watermk image', watermark.shape)

# 이미지 크기를 맞춰주기
img1 = cv2.resize(large_img, (800, 600))
img2 = cv2.resize(watermark, (800, 600))
# cv2.imshow('large', img1)
# cv2.imshow('watermk', img2)
# cv2.waitKey(0)

# 이미지 블렌딩
blend = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
cv2.imshow('blend', blend)
cv2.waitKey(0)
