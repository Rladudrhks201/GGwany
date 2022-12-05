import numpy as np

# 실행마다 동일한 결과를 얻기 위해 Seed를 생성
np.random.seed(3241)
print(np.random.randint(0,10,(2,3)))