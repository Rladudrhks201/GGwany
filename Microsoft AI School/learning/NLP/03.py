from nltk.stem.porter import PorterStemmer

# 단어 토큰 만들기
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# 어간 추출기를 만들기
porter = PorterStemmer()

# 어간 추출기를 적용
word_list = [porter.stem(word) for word in tokenized_words]
print(word_list)


