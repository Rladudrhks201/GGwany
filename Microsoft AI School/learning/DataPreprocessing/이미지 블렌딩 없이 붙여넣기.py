# 이미지 붙여넣기
import cv2

large_img = cv2.imread('./tennis.png')
watermark = cv2.imread('./goat.png')

small_img = cv2.resize(watermark, (300, 300))

x_offset, y_offset = 30, 170
x_end = x_offset + small_img.shape[0]
y_end = y_offset + small_img.shape[1]

large_img2 = large_img.copy()
large_img2[y_offset:y_end, x_offset:x_end] = small_img
cv2.imshow('paste', large_img2)
cv2.waitKey(0)
