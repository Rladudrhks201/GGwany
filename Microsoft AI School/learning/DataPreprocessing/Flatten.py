import cv2
from utils import image_show

image_path = './images.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image_10x10 = cv2.resize(image, (10, 10))
image_10x10.flatten() # 1차원 벡터로 변환
image_show(image_10x10)

# another case
image = cv2.imread(image_path)
# image 10x10 픽셀 크기로 변환
image_color_10x10 = cv2.resize(image, (10, 10))
image_color_10x10.flatten()
image_show(image_color_10x10)

# image 225x255 픽셀 크기로 변환
image_color_225x255 = cv2.resize(image, (225, 255))
image_color_225x255.flatten()
image_show(image_color_225x255)