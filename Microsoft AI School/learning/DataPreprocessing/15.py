import pandas as pd

data = pd.Timestamp('2022-12-05 01:40:00')  # dateitem 생성

data_in_london = data.tz_localize(tz='Europe/London')
print(data_in_london)

data_in_london.tz_convert('Africa/Abidjan')
# 시간대 변경
dates = pd.Series(pd.date_range('2/2/2022', periods=3, freq='M'))
temp = dates.dt.tz_localize('Africa/Abidjan')
