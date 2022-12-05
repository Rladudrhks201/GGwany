# 날짜 간의 차이 계산
import pandas as pd

dataframe = pd.DataFrame()
dataframe['Arrived'] = [pd.Timestamp('01-01-2022'), pd.Timestamp('01-04-2022')]
dataframe['Left'] = [pd.Timestamp('01-01-2022'), pd.Timestamp('01-06-2022')]


data = pd.Series(delta.days for delta in (
    dataframe['Left'] - dataframe['Arrived']
))
print(data)
# 두 지점사이의 시간차를 계산