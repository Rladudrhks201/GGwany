import numpy as np

# 단일 객체 저장 및 불러오기
array = np.arange(0, 10)
print(array)

# .npy 파일에 저장하기
np.save('./save.npy', array)

# 불러오기
result = np.load('./save.npy')
print('result = ', result)

# 복수 개체 저장을 위해 생성
array1 = np.arange(0, 10)
array2 = np.arange(0, 20)

# 저장
np.savez('./save.npz', array1 = array1, array2 = array2)

# 불러오기
data = np.load('./save.npz')
result1 = data['array1']
result2 = data['array2']

print(result1)
print(result2)