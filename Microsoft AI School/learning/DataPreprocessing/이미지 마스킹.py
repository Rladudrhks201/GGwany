import cv2
import numpy as np

mask = np.zeros((683, 1024), dtype=np.uint8)
cv2.rectangle(mask, (60, 50), (280, 280), (255, 255, 255), -1)
cv2.rectangle(mask, (480, 50), (610, 230), (255, 255, 255), -1)
cv2.rectangle(mask, (750, 50), (920, 280), (255, 255, 255), -1)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)

tennis = cv2.imread('./tennis.png')
tennis_gray = cv2.cvtColor(tennis, cv2.COLOR_BGR2GRAY)
bg = cv2.bitwise_and(tennis, tennis, mask=mask)
cv2.imshow('test', bg)
cv2.waitKey(0)


