# 딕셔너리 특성 행렬 변환
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

data_dict = [{"Red": 2,"Blue": 4},
             {"Red": 4,"Blue": 3},
             {"Red": 1,"Yellow": 2},
             {"Red": 2,"Yellow": 2}]

dictvectorizer = DictVectorizer(sparse=False)
# 0아닌 원소만 저장하는 희소 행렬을 반환
# 딕셔너리를 특성 행렬로 변환
features = dictvectorizer.fit_transform(data_dict)
features_name = dictvectorizer.get_feature_names()
print(features_name)
dict_data = pd.DataFrame(features, columns=features_name)
print(dict_data)

## 행렬 형태
# 단어 카운트 딕셔너리를 만들기
doc_1_word_count = {"Red": 2,"Blue": 4}
doc_2_word_count = {"Red": 4,"Blue": 3}
doc_3_word_count = {"Red": 1,"Yellow": 2}
doc_4_word_count = {"Red": 2,"Yellow": 2}
doc_word_counts = [doc_1_word_count,doc_2_word_count,doc_3_word_count,doc_4_word_count]
print(doc_word_counts)

data_array = dictvectorizer.fit_transform(doc_word_counts)
print(data_array)