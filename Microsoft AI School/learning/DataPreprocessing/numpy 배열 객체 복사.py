import numpy as np

array1 = np.arange(0, 10)
array2 = array1.copy()
array2[0] = 333
# 복사를 쓰지 않으면 원본도 같은 주소를 공유하기 때문에 원본도 변경
print(array1)
print(array2)

