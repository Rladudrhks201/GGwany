from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os

options = webdriver.ChromeOptions()
options.add_experimental_option('detach', True)
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(3)
driver.get('https://www.google.com')

keywords = '사과'
elem = driver.find_element(By.NAME, 'q')
elem.send_keys(keywords)
elem.send_keys(Keys.RETURN)  # 엔터 입력

driver.find_element(By.XPATH, '/html/body/div[7]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a').click()
# 이미지 클릭

# 스크롤
elem = driver.find_element(By.TAG_NAME, 'body')
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)

try:
    driver.find_element(By.XPATH,
                        '/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
except:
    pass

for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)

# 이미지 가져오기
links = []
images = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i.Q4LuWd')  # 클래스로 가져오기

for image in images:
    if image.get_attribute('src') != None:
        links.append(image.get_attribute('src'))

print(len(links))

# 폴더 생성
os.makedirs('C:/Users/user/Desktop/Search/Data', exist_ok=True)

# 이미지 데이터 다운로드

for index, i in enumerate(links):
    url = i
    start = time.time()
    urllib.request.urlretrieve(url, 'C:/Users/user/Desktop/Search/Data/' +
                               keywords + '_' + str(index) + '.jpg')
    print(str(index) + "/" + str(len(links)) + " " + keywords +
          " 다운로드 시간 ------ : ", str(time.time() - start)[:5] + '초')

print(keywords + "다운로드 완료 !!")
