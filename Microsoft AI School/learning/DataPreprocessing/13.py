# 텍스트 데이터 처리
import re
text_data = ["my name is Kim Yeong Gwan,",
             "  My height is 185cm.",
             "    And my weight is 82kg."]

# 공백 제거
strip_whitespace = [string.strip() for string in text_data]
print(strip_whitespace)

# 마침표 제거
remove_periods = [string.replace(".","") for string in strip_whitespace]
print(remove_periods)

def capitalizer(string: str) -> str: return string.upper()



temp = [capitalizer(string) for string in remove_periods]
print(temp)



