# 순서가 없는 범주형 데이터 처리
# 사이킷런의 LabelBinarize를 사용하여 문자열 타깃 데이터 원 핫 인코딩 진행
import numpy as np
from sklearn.preprocessing import OneHotEncoder

feature = np.array([["texas",1],
                    ["california",1],
                    ["texas",3],
                    ["delaware",1],
                    ["texas",1]])
print(feature)
one_hot_encoder = OneHotEncoder(sparse=False)
# 숫자도 범주형 변수로 취급
print(one_hot_encoder.fit_transform(feature))
print(one_hot_encoder.categories_)
# 여러 개의 열이 있는 특성 배열 생성
