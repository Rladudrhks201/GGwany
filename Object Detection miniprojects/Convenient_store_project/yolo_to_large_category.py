import os
import glob
import pandas as pd
from collections import deque
from tqdm import tqdm

final = pd.read_csv('.\\final.csv', encoding='utf-8-sig')
finalidx = list(final.index)
standard = pd.read_csv('.\\Large_label_standard.csv', encoding='utf-8-sig')
standard.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n']
stdlist = list(set(list(standard['div_l'])))
stdlist = deque(stdlist)
stdlist.remove('이/미용')
stdlist.remove('생활용품')
stdlist.remove('소스')
stdlist = list(stdlist)
stdlist.sort()
# print(stdlist)
# ['과자', '디저트', '면류', '상온HMR', '유제품', '음료', '주류', '커피차', '통조림/안주']


paths = glob.glob(os.path.join('D:\\dataset01', '*', 'labels', '*.txt'))
paths = paths[:352475]
for path in tqdm(paths):
    tr_vl = path.split('\\')[-3]
    filename = os.path.basename(path)
    labels = []
    yoloxs = []
    yoloys = []
    yolows = []
    yolohs = []
    code = []
    errors = 0
    with open(path, 'r') as f:
        for i in f.readlines():
            if errors != 0:
                continue
            label = i.split(' ')[0]
            yolox = i.split(' ')[1]
            yoloy = i.split(' ')[2]
            yolow = i.split(' ')[3]
            yoloh = i.split(' ')[4]
            # print(label, yolox, yoloy, yolow, yoloh)
            lb = final[final.index == int(label)]['code']
            lb = str(list(lb)[0])
            code.append(lb)
            new_label = standard[standard['code'] == lb]['div_l']
            new_label = list(new_label)[0]
            try:
                new_label = stdlist.index(new_label)
            except:
                errors += 1
                continue
            labels.append(new_label)
            yoloxs.append(yolox)
            yoloys.append(yoloy)
            yolows.append(yolow)
            yolohs.append(yoloh)
    os.makedirs(f'D:\\Large_label\\{tr_vl}', exist_ok=True)
    os.makedirs(f'D:\\Large_label\\{tr_vl}\\labels', exist_ok=True)

    for new_label, yolox, yoloy, yolow, yoloh in zip(labels, yoloxs, yoloys, yolows, yolohs):
        os.makedirs(f'D:\\Large_label\\{tr_vl}\\labels\\{new_label}', exist_ok=True)
        os.makedirs(f'D:\\Large_label\\{tr_vl}\\labels\\{new_label}\\{code[0]}', exist_ok=True)
        with open(os.path.join('D:\\Large_label', f'{tr_vl}', 'labels', f'{new_label}', f'{code[0]}', f'{filename}'),
                  'a') as f:
            f.write(f'{new_label} {yolox} {yoloy} {yolow} {yoloh} \n')
