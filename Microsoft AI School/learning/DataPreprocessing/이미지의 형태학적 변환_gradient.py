# Gradient : 모서리를 감지, dilation - erosion
# tophat : 밝은 부분, original - opening
# blackhat : 어두운 부분, closing - original
# opening : 노이즈 제거, erosion -> dilation, dilation@erosion
# closing : 개체의 바운더리, dilation -> erosion, erosion@dilation

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

op_idx = {
    'gradient': cv2.MORPH_GRADIENT,
    'tophat': cv2.MORPH_TOPHAT,
    'blackhat': cv2.MORPH_BLACKHAT
}


def onchange(k, op_name):
    if k == 0:
        cv2.imshow(op_name, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dst = cv2.morphologyEx(mask, op_idx[op_name], kernel)
    cv2.imshow(op_name, dst)


cv2.imshow('original', img)
cv2.imshow('gradient', mask)
cv2.imshow('tophat', mask)
cv2.imshow('blackhat', mask)

cv2.createTrackbar('k', 'gradient', 0, 300, lambda x: onchange(k=x, op_name='gradient'))
cv2.createTrackbar('k', 'tophat', 0, 300, lambda x: onchange(k=x, op_name='tophat'))
cv2.createTrackbar('k', 'blackhat', 0, 300, lambda x: onchange(k=x, op_name='blackhat'))

cv2.waitKey(0)
cv2.destroyAllWindows()