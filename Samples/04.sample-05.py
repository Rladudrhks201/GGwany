from email import header
from typing import Container
from urllib import request, response
from currency_converter import CurrencyConverter

cc = CurrencyConverter('http://www.ecb.europa.eu/stats/eurofxref/eurofxref.zip')
print(cc.currencies)
print(cc.convert(1,'USD','KRW')) #USD 기준으로 원화의 가치

import requests
from bs4 import BeautifulSoup

def get_exchange_rate(target1,target2):
    headers = {
        'User-Agent':'Mozilla/5.0',
        'Content-Type':'text/html; charset=utf-8'
    }
    responses = requests.get(f'https://kr.investing.com/currencies/{target1}-{target2}',headers=headers)
    content = BeautifulSoup(response.content,'html.parser')
    Containers = content.find('span',{'data-test':'instrument-price-last'})

    print(Containers.text)

get_exchange_rate('usd','krw')

