# 배경 제거
import numpy as np
from utils import image_show
import cv2
import matplotlib.pyplot as plt

# 이미지 경로 및 읽기
image_path = './test.png'
image = cv2.imread(image_path)

# 사각형 좌표 : 시작점: x , y , 넓이 , 높이
# image = cv2.rectangle(image, (400, 100), (30, 10), (255, 0, 255), 2)
# image_show(image)
rectangle = (0, 0, 400, 400)
# rectangle = cv2.selectROI(image)
# 초기 마스크 생성
mask = np.zeros(image.shape[:2], np.uint8)

# grabcut에 사용할 임시 배열 생성
bgdmodel = np.zeros((1, 65), np.float64)
fgdmodel = np.zeros((1, 65), np.float64)

# grabcut 실행
# image = 원본이미지, bgdmodel = 배경을 위한 임시 배열, fgdmodel = 전경배경, 5 반복회수, cv2.GC_INIT_WITH_RECT 사각형 초기화
cv2.grabCut(image, mask, rectangle, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)

# 배경인 곳은 0 그 외에는 1로 설정한 마스크 생성
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크 곱해서 -> 배경 제외
image_rgb_nobg = image * mask_2[:, :, np.newaxis]
image_show(image_rgb_nobg)