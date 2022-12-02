# 순서가 있는 범주형 특성 인코딩
import pandas as pd

data = {"score": ["Low","Low","Medium","High"]}
dataframe = pd.DataFrame(data)
print(dataframe)

# 매핑 딕셔너리 생성
scale_mapper = {"Low": 1, "Medium": 2, "High": 3}
dataframe['score'] = dataframe['score'].replace(scale_mapper)
print(dataframe)