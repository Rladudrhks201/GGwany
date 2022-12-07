# 침식, erosion
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

# 우리가 사용하는 추가 매개변수는 임계 강도 픽셀 값이며 이 경우에는 230과 255값으로 설정,
# 230보다 큰 모든 값이 255로 설정됨을 의미합니다
# 단순히 값을 반전시키는 임계값 알고리즘 THRESH_BINARY_INV (230보다 작으면 흰색, 230보다 큰 값은 검은색)

_, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)

# erosion , 검은 면적이 더 커짐 (침식)
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(mask, kernel)
erosion = cv2.erode(mask, kernel)

titles = ['image', 'mask', 'dilation', 'erosion']
images = [image, mask, dilation, erosion]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

# 커널 종류, 더 많은 종류가 있다
kernel = []
for i in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
    kernel.append(cv2.getStructuringElement(i, (11,11)))
print([kernel[i] for i in range(3)]) # 전부 1, 크로스, 원형 형태

erosion1 = cv2.erode(mask, kernel[0])
erosion2 = cv2.erode(mask, kernel[1])
erosion3 = cv2.erode(mask, kernel[2])

titles = ['image', 'rectangle', 'cross', 'ellipse']
images = [image, erosion1, erosion2, erosion3]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

