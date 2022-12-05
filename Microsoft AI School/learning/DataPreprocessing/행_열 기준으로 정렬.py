import numpy as np

array = np.array([[5, 2, 7, 6], [2, 3, 14, 8]])
print(array)
# 각 열 기준으로 정렬 후
array.sort(axis=0)
print(array)

# 각 행 기준으로 정렬 후
array.sort(axis=1)
print(array)