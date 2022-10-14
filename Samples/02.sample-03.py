from gtts import gTTS
from playsound import playsound
import os 

os.chdir(os.path.dirname(os.path.abspath(__file__))) # os의 change directory

text = "안녕하세요 마이크로소프트 에이아이 스쿨에 오신 것을 환영합니다."

tts = gTTS(text = text, lang='ko')

current_file_path = os.getcwd() + "./hi.mp3"
tts.save(current_file_path)

playsound(current_file_path)

# TTS를 생성하는 라이브러리를 이용하여 음성파일을 만들고 재생