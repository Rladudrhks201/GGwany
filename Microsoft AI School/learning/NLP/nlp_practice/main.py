# pip install transformers
from transformers import pipeline
from transformers import set_seed
import pandas as pd

set_seed(888)
classifier = pipeline('text-classification')
text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# 처음 이 코드를 실행하면 파이프라인이 자동으로 허깅페이스 허브에서 모델 가중치 다운로드 합니다.
# 파이프라인 객체를 다시 만들 때는 가중치가 이미 다운로드됐으므로 캐싱된 버전을 사용한다는 안내 메시지가 나옵니다.
# 기본적으로 txt-classification 파이프라인은 감성 분석을 위해 설계된 모델을 사용하지만, 다중 분류와 다중 레이블 분류도 지원합니다.

outputs = classifier(text)
print(outputs)
# 긍정, 부정 평가

ner_tagger = pipeline('ner', aggregation_strategy='simple')
outputs = ner_tagger(text)  # 집합화
temp = pd.DataFrame(outputs)
print(temp)
# 그룹화

reder = pipeline('question-answering')
question = 'what does the customer want?'
outputs = reder(question=question, context=text)
outputs = pd.DataFrame([outputs])
print(outputs)
# 질문 답변

summarizer = pipeline('summarization')
outputs = summarizer(text, max_length=60, clean_up_tokenization_spaces=True)
# print(outputs[0]['summary_text'])
print(outputs)
# 요약

translater = pipeline('translation', model='Helsinki-NLP/opus-mt-ja-en')
print(translater("愚か者"))
# 번역, Low Quality

generator = pipeline('text-generation')
response = 'Dear Bumblebee, I am sorry to hear that your order was mixed up'
prompt = text + '\n\nCustomer service response : \n' + response
outputs = generator(prompt, max_length=400)
print(outputs[0]['generated_text'])