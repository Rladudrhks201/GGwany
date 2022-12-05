import numpy as np

# 중복 원소 제거
array = np.array([1, 2, 1, 2, 3, 4, 3, 4, 5])
print('중복 처리 전', array)
print('중복 처리 후', np.unique(array))

# np.isin() : 찾는 원소가 있는지 여부를 각 index 위치에 true false 값으로 반환해줌
array = np.arange(1,8)
iwantit = np.array([1,2,3,10])
print(np.isin(array, iwantit))
"""
[ True  True  True False False False False]
"""
