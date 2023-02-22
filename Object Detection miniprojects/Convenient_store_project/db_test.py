import pymysql
import glob
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 라벨 불러오기
large = {}
beverage = {}
drink = {}
hmr = {}
snack = {}
coffee_tea = {}
canned_food = {}
dessert_noodle_dairy = {}
lbs_list = ['과자', '디저트', '면류', '상온HMR', '유제품', '음료', '주류', '커피차', '통조림/안주']
lb_dict = {}
for i, lb in enumerate(lbs_list):
    lb_dict[lb] = str(i)

id_paths = glob.glob(os.path.join('C:\\Users\\fiter\\Desktop\\codezip', '*.txt'))
for path in id_paths:
    name = os.path.basename(path)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').strip()
            no = line.split(':')[0]
            code = line.split(':')[1].strip()
            if name == 'beverage.txt':
                beverage[code] = no
            elif name == 'dessert_noodle_dairy.txt':
                dessert_noodle_dairy[code] = no
            elif name == 'drink.txt':
                drink[code] = no
            elif name == 'hmr.txt':
                hmr[code] = no
            elif name == 'snack.txt':
                snack[code] = no
            elif name == 'large.txt':
                large[code] = no
            elif name == 'coffee_tea.txt':
                coffee_tea[code] = no
            elif name == 'canned_food.txt':
                canned_food[code] = no
            else:
                print('딕셔너리 생성 오류', path)
                exit()



#데이터베이스 연결 전 데이터 기본 세팅
host_name = 'localhost'
port = 3306
user_name = 'root'
user_password = '비번'                     # <-------- 이거 비번 저랑 다르시면 수정 필요해요
database_name = 'product_detection'

test_db = pymysql.connect( 
    host = host_name,
    port = port,
    user = user_name,
    passwd = user_password,
    db = database_name,
    charset = 'utf8'
)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 데이터 INSERT ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
cursor = test_db.cursor()


# SQL 커맨드(INSERT INTO "TABLE NAME" (COLUMNS) VALUES (DATA))   <- 여기서 DB 보낼때 %s로 하여도 db에서는 알아서 int형으로 저장되어진다.
# 대소세 테이블에 데이터를 무조건 먼저 입력하여야 됩니다!! FK(외래키) 때문에 특정 튜플이 없으면 label 테이블에 튜플을 삽입을 못 해요

# 각 테이블에 필요한 쿼리문
sql_div_large = "INSERT INTO div_large (div_l_no, div_l_name) VALUES (%s, %s)"
sql_div_detail = "INSERT INTO div_detail (div_n_no, div_l_label, div_n_name) VALUES (%s, %s, %s)"
sql_label = "INSERT INTO label(item_no, barcd, prod_nm, div_n_no, volume, nutrition_info) VALUES (%s, %s, %s, %s, %s, %s)"


xml_path = glob.glob(os.path.join('C:\\Users\\fiter\\Desktop\\final', '*.xml'))
divl = []
divl2 = []
divn = []
for path in tqdm(xml_path):
    tree = ET.parse(path)
    root = tree.getroot()
    element = root.find('div_cd')
    item_no = element.find('item_no').text
    barcd = element.find('barcd').text
    prod_nm = element.find('img_prod_nm').text
    div_l_name = element.find('div_l').text
    try:
        div_l_no = str(lb_dict[div_l_name])
    except:
        print(div_l_name, '대분류 라벨 생성 오류')
    if div_l_no == '0':
        div_n_no = str(snack[str(item_no)])
    elif div_l_no == '1':
        div_n_no = str(dessert_noodle_dairy[str(item_no)])
    elif div_l_no == '2':
        div_n_no = str(dessert_noodle_dairy[str(item_no)])
    elif div_l_no == '3':
        div_n_no = str(hmr[str(item_no)])
    elif div_l_no == '4':
        div_n_no = str(dessert_noodle_dairy[str(item_no)])
    elif div_l_no == '5':
        div_n_no = str(beverage[str(item_no)])
    elif div_l_no == '6':
        div_n_no = str(drink[str(item_no)])
    elif div_l_no == '7':
        div_n_no = str(coffee_tea[str(item_no)])
    elif div_l_no == '8':
        div_n_no = str(canned_food[str(item_no)])
    else:
        print('소분류 생성 오류', div_l_no)
    div_n_name = element.find('div_n').text
    volume = element.find('volume').text
    nutrition_info = element.find('nutrition_info').text.strip('{').strip('}')
    # print(item_no, barcd, prod_nm, div_l_no, div_l_name, div_n_no, div_n_name, volume, nutrition_info)
    # exit()
    if div_l_no not in divl:
        cursor.execute(sql_div_large, (div_l_no, div_l_name))
        divl.append(div_l_no)
    if div_n_no not in divn:
        cursor.execute(sql_div_detail, (div_n_no, div_l_no, div_n_name))
        divl2.append(div_l_no)
        divn.append(div_n_no)
    elif div_l_no not in divl2:
        cursor.execute(sql_div_detail, (div_n_no, div_l_no, div_n_name))
        divl2.append(div_l_no)
        divn.append(div_n_no)
    try:
        cursor.execute(sql_label, (item_no, barcd, prod_nm, div_n_no, volume, nutrition_info))
    except:
        print(path)

test_db.commit()
test_db.close()
