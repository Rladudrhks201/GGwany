# 불용어 삭제
from nltk.corpus import stopwords
import nltk

# 불용어 데이터를 다운로드
nltk.download('stopwords')

# 단어 토큰을 생성
tokenizd_words = ['i', 'am', 'the', 'of', 'to', 'go', 'store', 'and', 'park']
# ctrl alt L 로 자동 문법 정렬

# 불용어 로드
stop_words = stopwords.words('english')
print(len(stop_words))

# 불용어 삭제
[word for word in tokenizd_words if word not in stop_words]

stop_data = stop_words
print(stop_data)
