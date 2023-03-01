import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np


# 한글 폰트
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/NGULIM.TTF'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

df = pd.read_csv('.\\combined_data2.csv', encoding='utf-8-sig')
# 날짜 타입으로 변환
df['일자'] = df['일자'].astype('str')
df['일자'] = pd.to_datetime(df['일자'])
# df.info()

df = df.iloc[:, 1:]
# df = df[df['일자'] < datetime.strptime('20230225', '%Y%m%d')]   # 마지막 관측값 까지 -> 2612
df = df[df.index < 2612]
df = df.fillna(0)
df = df.replace(0, np.NaN)
cols0 = ['건고추_가격(원/kg)', '깻잎_가격(원/kg)',
       '당근_가격(원/kg)', '대파_가격(원/kg)',
       '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
       '배추_가격(원/kg)', '백다다기_가격(원/kg)',
       '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
       '시금치_가격(원/kg)', '애호박_가격(원/kg)',
       '양배추_가격(원/kg)', '양파_가격(원/kg)',
       '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
       '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
       '팽이버섯_가격(원/kg)', '포도_가격(원/kg)']
for col in cols0:
       df[col] = df[col].interpolate(method='linear').fillna(0)


df['weekday'] = df['일자'].dt.weekday
weekday_list = ['월', '화', '수', '목', '금', '토', '일']
df['요일'] = df.apply(lambda x : weekday_list[x['weekday']], axis=1)
# 0은 월요일 6은 일요일

df = pd.concat([df, pd.get_dummies(df['요일'])], axis=1)
df = df[['일자', '요일', '건고추_가격(원/kg)', '깻잎_가격(원/kg)',
       '당근_가격(원/kg)', '대파_가격(원/kg)',
       '마늘_가격(원/kg)', '무_가격(원/kg)', '미나리_가격(원/kg)',
       '배추_가격(원/kg)', '백다다기_가격(원/kg)',
       '새송이_가격(원/kg)', '샤인머스캇_가격(원/kg)',
       '시금치_가격(원/kg)', '애호박_가격(원/kg)',
       '양배추_가격(원/kg)', '양파_가격(원/kg)',
       '얼갈이배추_가격(원/kg)', '청상추_가격(원/kg)',
       '토마토_가격(원/kg)', '파프리카_가격(원/kg)',
       '팽이버섯_가격(원/kg)', '포도_가격(원/kg)', '월', '화',
       '수', '목', '금', '토', '일']]
# print(df)




# feature, target 설정
feature = df.columns[2:]
# feature = ['배추_거래량(kg)', '배추_가격(원/kg)', '월', '화', '수', '목', '금', '토', '일']   # 성능이 더 낮음
# 예측할 작물의 가격의 8일치를 target에 설정
df['target'] = df['배추_가격(원/kg)'].shift(-8)

df_learn = df[df['target'].notnull()]
X = df_learn[feature]
y = df_learn['target']


# 두 번째 탐색적 모델 (랜덤포레스트, XGBoost 사용)
model = RandomForestRegressor()
model.fit(X, y)

y_pred = model.predict(X)
# R2 = model.score(X_train, y_train)
MAE = mean_absolute_error(y, y_pred)

plt.figure(figsize=(20, 10), dpi=300)
plt.title('RandomForest 예측 결과'+   '   MAE : ' + str(MAE)[:7])
plt.ylabel('가격')
plt.plot(np.array(y), alpha = 0.9, label = 'Real')
plt.plot(model.predict(X), alpha = 0.6, linestyle = "--", label = 'Predict')
plt.legend()
plt.show()

# XGBoost
model = XGBRegressor()
model.fit(X, y)

y_pred = model.predict(X)
# R2 = model.score(X_train, y_train)
MAE = mean_absolute_error(y, y_pred)

plt.figure(figsize=(20, 10), dpi=300)
plt.title('XGBoost 예측 결과'+   '   MAE : ' + str(MAE)[:7])
plt.ylabel('가격')
plt.plot(np.array(y), alpha = 0.9, label = 'Real')
plt.plot(model.predict(X), alpha = 0.6, linestyle = "--", label = 'Predict')
plt.legend()
plt.show()









