import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

# 우리가 사용하는 추가 매개변수는 임계 강도 픽셀 값이며 이 경우에는 230과 255값으로 설정,
# 230보다 큰 모든 값이 255로 설정됨을 의미합니다
# 단순히 값을 반전시키는 임계값 알고리즘 THRESH_BINARY_INV (230보다 작으면 흰색, 230보다 큰 값은 검은색)

_, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)

# open : erosion -> dilation (to delete dot noise), 이미지를 샤프하게 만든후 dilation으로 샤프한 노이즈를 제거
# 원본의 데이터는 보존하면서 노이즈를 제거
# close : dilation -> erosion
kernel = np.ones((3, 3), np.uint8)

erosion = cv2.erode(mask, kernel, iterations=1)
opening = cv2.dilate(erosion, kernel, iterations=1)

plt.subplot(1, 2, 1)
plt.imshow(opening, 'gray')
plt.title('manual opening')

f_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
plt.subplot(1, 2, 2)
plt.imshow(f_opening, 'gray')
plt.title('opening')
plt.show()
