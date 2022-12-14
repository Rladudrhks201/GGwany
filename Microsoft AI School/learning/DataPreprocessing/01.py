import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 정규화, 표준화 

def normalization(data):
    data_mm = (data - data.min(axis=0))/ (data.max(axis=0) - (data.min(axis=0)))
    return data_mm

def numpy_standardization(data):
    """
    (각 데이터 - 평균(각열)) / (표준편차(각열))
    """
    std_data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
    return std_data

def main():
    data = np.random.randint(30,size=(6,5))
    print(data)
    std_data_temp = numpy_standardization(data)
    print(std_data_temp)

    no_data = normalization(data)
    print(no_data)


if __name__ == '__main__':
    main()
# 인터프리터에서 직접 실행했을때만 실행, import py에서 사용하면 실행안됨