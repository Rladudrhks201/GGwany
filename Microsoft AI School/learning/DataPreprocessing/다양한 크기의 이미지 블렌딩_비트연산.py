import cv2
import numpy as np

# ex1
img_rectangle = np.ones((400, 400), dtype=np.uint8)
cv2.rectangle(img_rectangle, (50, 50), (300, 300), (255, 255, 255), -1)  # 넣을 이미지, 사각형크기, 위치, 색상 ,가득
# cv2.imshow('image show', img_rectangle)
# cv2.waitKey(0)

# ex2
img_circle = np.ones((400, 400), dtype=np.uint8)
cv2.circle(img_circle, (300, 300), 70, (255, 255, 255), -1)
# cv2.imshow('circle', img_circle)
# cv2.waitKey(0)

# ex3
bitwiseAnd = cv2.bitwise_and(img_rectangle, img_circle)
# cv2.imshow("bitwise_and", bitwiseAnd)
# cv2.waitKey(0)
# 겹친 부분만 나옴

bitwiseOr = cv2.bitwise_or(img_rectangle, img_circle)
# cv2.imshow("bitwise_or", bitwiseOr)
# cv2.waitKey(0)

bitwiseXor = cv2.bitwise_xor(img_rectangle, img_circle)
# cv2.imshow("bitwise_xor", bitwiseXor)
# cv2.waitKey(0)

rec_not = cv2.bitwise_not(img_rectangle)
# cv2.imshow('rectangle not', rec_not)
# cv2.waitKey(0)

# ex4 마스킹 과제, 흰색 대신 이미지를 넣기
mask = np.zeros((683, 1024), dtype=np.uint8)
cv2.rectangle(mask, (60, 50), (280, 280), (255, 255, 255), -1)
cv2.rectangle(mask, (420, 50), (550, 230), (255, 255, 255), -1)
cv2.rectangle(mask, (750, 50), (920, 280), (255, 255, 255), -1)
cv2.imshow('mask', mask)
cv2.waitKey(0)