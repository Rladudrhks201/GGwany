import numpy as np
import cv2

# 이미지 경로
x = cv2.imread('./images (2).jpeg', 0)  # 흑백 이미지
y = cv2.imread('./images (2).jpeg', 1)  # 컬러 이미지

# cv2.resize(x, (900,900)) 이미지 크기 조정
cv2.imshow("dog image show gray", x)
cv2.imshow("dog image show", y)
cv2.waitKey(0)

# 여러개 파일 save .npz
np.savez('./image.npz', array1 = x, array2 = y)

# 압축 방법
np.savez_compressed('./image_compressed.npz', array1 = x, array2 = y)

# npz 데이터 로드
data = np.load('./image_compressed.npz')

result1 = data['array1']
result2 = data['array2']

cv2.imshow("result1", result1)
cv2.imshow("result2", result2)
cv2.waitKey(0)  # 비디오인 경우 1
