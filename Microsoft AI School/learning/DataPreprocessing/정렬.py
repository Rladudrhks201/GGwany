# Numpy 가장 많이 사용되는 함수
# 1. 원소 정렬
import numpy as np

# default -> 오름차순
array = np.array([15, 20, 5, 12, 7])
np.save('./array.npy', array)

array_data = np.load('./array.npy')
print(array_data)
array_data.sort()
print(array_data) # 오름차순 정렬
print(array_data[::-1]) # 내림차순 정렬

