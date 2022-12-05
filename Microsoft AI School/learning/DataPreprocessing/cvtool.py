import cv2
import numpy as np

img = cv2.imread('./images (2).jpeg')

print('image의 타입 : ', type(img))
print('image의 크기 : ', img.shape)

"""
image의 타입 :  <class 'numpy.ndarray'>
image의 크기 :  (275, 183, 3)
"""
h, w, _ = img.shape
print(h, w)

cv2.imshow("image show", img)
cv2.waitKey(0)