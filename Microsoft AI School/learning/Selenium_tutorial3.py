from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os
from multiprocessing import Pool
import pandas as pd


def create_folder(directory):
    os.makedirs(directory, exist_ok=True)


# 검색 키워드 호출
key = pd.read_csv('C:/Users/user/Desktop/Search/Data/Keyword.txt', encoding='utf-8', names=['keyword'])

keyword = []
[keyword.append(key['keyword'][i]) for i in range(len(key))]
print(keyword)


def image_download(keyword):
    create_folder('C:/Users/user/Desktop/Search/Data/' + keyword + 'high_res')

    options = webdriver.ChromeOptions()
    options.add_experimental_option('detach', True)
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(3)
    driver.get('https://www.google.com')

    keywords = keyword
    elem = driver.find_element(By.NAME, 'q')
    elem.send_keys(keywords)
    elem.send_keys(Keys.RETURN)  # 엔터 입력

    driver.find_element(By.XPATH, '/html/body/div[7]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a').click()

    elem = driver.find_element(By.TAG_NAME, 'body')
    for i in range(100):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.4)

    try:
        driver.find_element(By.XPATH,
                            '/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[2]/div[2]/input').click()
    except:
        pass

    for i in range(100):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.4)

    links = []
    images = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i.Q4LuWd')  # 클래스로 가져오기

    for i in range(1, len(images)):
        try:


            driver.find_element(By.XPATH, f'//*[@id="islrg"]/div[1]/div[{i}]/a[1]/div[1]/img').click()
            links.append(driver.find_element(By.XPATH, '//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/di'
                                                       'v[2]/div[1]/div[1]/div[2]/div/a/img').get_attribute('src'))

            if len(links) == 200:  # 너무 오래 걸림
                break
        except:
            continue
    print(keywords + '찾은 이미지 개수 : ', len(links))
    time.sleep(2)

    forbidden = 0

    for index, i in enumerate(links):
        try:
            url = i
            start = time.time()
            urllib.request.urlretrieve(url, 'C:/Users/user/Desktop/Search/Data/' + keywords + 'high_res/' +
                                       keywords + '_' + str(index) + '.jpg')
            print(str(index) + "/" + str(len(links)) + " " + keywords +
                  " 다운로드 시간 ------ : ", str(time.time() - start)[:5] + '초')
        except:
            forbidden += 1
            continue


if __name__ == '__main__':
    pool = Pool(processes=3)
    pool.map(image_download, keyword)
