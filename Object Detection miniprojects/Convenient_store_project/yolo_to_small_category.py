import os
import glob
import pandas as pd
from tqdm import tqdm
import shutil
import pickle

trigger = False
if trigger:
    final = pd.read_csv('.\\final.csv', encoding='utf-8-sig')
    final_idx = pd.Series(list(final.index))
    final = pd.concat([final, final_idx], axis=1)

    standard = pd.read_csv('.\\Large_label_standard.csv', encoding='utf-8-sig')
    standard.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n']
    snack = standard[standard['div_l'] == '과자']
    snack = snack.reset_index()
    snack_idx = pd.Series(list(snack.index))
    snack = pd.concat([snack, snack_idx], axis=1)
    snack = snack.iloc[:, 1:]
    snack.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']

    desert_noodle_dairy = standard[
        (standard['div_l'] == '디저트') | (standard['div_l'] == '면류') | (standard['div_l'] == '유제품')]
    desert_noodle_dairy = desert_noodle_dairy.reset_index()
    desert_noodle_dairy_idx = pd.Series(list(desert_noodle_dairy.index))
    desert_noodle_dairy = pd.concat([desert_noodle_dairy, desert_noodle_dairy_idx], axis=1)
    desert_noodle_dairy = desert_noodle_dairy.iloc[:, 1:]
    desert_noodle_dairy.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']

    hmr = standard[standard['div_l'] == '상온HMR']
    hmr = hmr.reset_index()
    hmr_idx = pd.Series(list(hmr.index))
    hmr = pd.concat([hmr, hmr_idx], axis=1)
    hmr = hmr.iloc[:, 1:]
    hmr.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']

    beverage = standard[standard['div_l'] == '음료']
    beverage = beverage.reset_index()
    beverage_idx = pd.Series(list(beverage.index))
    beverage = pd.concat([beverage, beverage_idx], axis=1)
    beverage = beverage.iloc[:, 1:]
    beverage.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']

    drink = standard[standard['div_l'] == '주류']
    drink = drink.reset_index()
    drink_idx = pd.Series(list(drink.index))
    drink = pd.concat([drink, drink_idx], axis=1)
    drink = drink.iloc[:, 1:]
    drink.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']

    coffee_tea = standard[standard['div_l'] == '커피차']
    coffee_tea = coffee_tea.reset_index()
    coffee_tea_idx = pd.Series(list(coffee_tea.index))
    coffee_tea = pd.concat([coffee_tea, coffee_tea_idx], axis=1)
    coffee_tea = coffee_tea.iloc[:, 1:]
    coffee_tea.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']

    canned_food = standard[standard['div_l'] == '통조림/안주']
    canned_food = canned_food.reset_index()
    canned_food_idx = pd.Series(list(canned_food.index))
    canned_food = pd.concat([canned_food, canned_food_idx], axis=1)
    canned_food = canned_food.iloc[:, 1:]
    canned_food.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels']
    standard = pd.concat([snack, desert_noodle_dairy, hmr, beverage, drink, coffee_tea, canned_food], axis=0)
    standard = standard.reset_index()
    standard = standard.iloc[:, 1:]

    std = pd.merge(left=standard, right=final, how='inner', on='code')
    std.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'labels', 'name', 'former_labels']
    # print(std)
    std.to_csv('.\\small_category.csv', encoding='utf-8-sig', index=False)

    # paths = glob.glob(os.path.join('D:\\data001', '*', 'labels', '*.txt'))
    # labels = []
    # for path in tqdm(paths):
    #     tr_vl = path.split('\\')[-3]
    #     filename = os.path.basename(path)
    #     txt_path = f"D:\\dataset01\\{tr_vl}\\labels\\{filename}"
    #     with open(txt_path, 'r') as f:
    #         for i in f.readlines():
    #             label = i.split(' ')[0]
    #             if label not in labels:
    #                 labels.append(label)
    # with open('.\\using_data.pkl', 'wb') as f:
    #     pickle.dump(labels, f)

    with open('.\\using_data.pkl', 'rb') as f:
        labels = pickle.load(f)

    labels = list(map(int, labels))

    snack = std[(std['former_labels'].isin(labels)) & (std['div_l'] == '과자')]
    snack = snack.reset_index()
    snack_idx = pd.Series(list(snack.index))
    snack = pd.concat([snack, snack_idx], axis=1)
    snack = snack.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    snack.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']


    hmr = std[(std['div_l'] == '상온HMR') & (std['former_labels'].isin(labels))]
    hmr = hmr.reset_index()
    hmr_idx = pd.Series(list(hmr.index))
    hmr = pd.concat([hmr, hmr_idx], axis=1)
    hmr = hmr.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    hmr.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']

    beverage = std[(std['div_l'] == '음료') & (std['former_labels'].isin(labels))]
    beverage = beverage.reset_index()
    beverage_idx = pd.Series(list(beverage.index))
    beverage = pd.concat([beverage, beverage_idx], axis=1)
    beverage = beverage.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    beverage.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']

    desert_noodle_dairy = std[
        (std['div_l'] == '디저트') | (std['div_l'] == '면류') | (std['div_l'] == '유제품')]
    desert_noodle_dairy = desert_noodle_dairy.reset_index()
    desert_noodle_dairy_idx = pd.Series(list(desert_noodle_dairy.index))
    desert_noodle_dairy = pd.concat([desert_noodle_dairy, desert_noodle_dairy_idx], axis=1)
    desert_noodle_dairy = desert_noodle_dairy.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    desert_noodle_dairy.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']

    drink = std[std['div_l'] == '주류']
    drink = drink.reset_index()
    drink_idx = pd.Series(list(drink.index))
    drink = pd.concat([drink, drink_idx], axis=1)
    drink = drink.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    drink.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']

    coffee_tea = std[std['div_l'] == '커피차']
    coffee_tea = coffee_tea.reset_index()
    coffee_tea_idx = pd.Series(list(coffee_tea.index))
    coffee_tea = pd.concat([coffee_tea, coffee_tea_idx], axis=1)
    coffee_tea = coffee_tea.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    coffee_tea.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']

    canned_food = std[std['div_l'] == '통조림/안주']
    canned_food = canned_food.reset_index()
    canned_food_idx = pd.Series(list(canned_food.index))
    canned_food = pd.concat([canned_food, canned_food_idx], axis=1)
    canned_food = canned_food.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    canned_food.columns = ['code', 'div_l', 'div_m', 'div_s', 'div_n', 'name', 'former_labels', 'labels']


    std = pd.concat([snack, desert_noodle_dairy, hmr, beverage, drink, coffee_tea, canned_food], axis=0)
    std = std.reset_index()
    std = std.iloc[:, 1:]

    std.to_csv('.\\small_category.csv', encoding='utf-8-sig', index=False)

    exit()

strd = pd.read_csv('.\\small_category.csv', encoding='utf-8-sig')

# ['과자', '디저트', '면류', '상온HMR', '유제품', '음료', '주류', '커피차', '통조림/안주']
lg_dict = {}
lg_dict['과자'] = 'snack'
lg_dict['디저트'] = 'dessert_noodle_dairy'
lg_dict['면류'] = 'dessert_noodle_dairy'
lg_dict['유제품'] = 'dessert_noodle_dairy'
lg_dict['상온HMR'] = 'hmr'
lg_dict['음료'] = 'beverage'
lg_dict['주류'] = 'drink'
lg_dict['커피차'] = 'coffee_tea'
lg_dict['통조림/안주'] = 'canned_food'

for i in lg_dict.values():
    os.makedirs(f"D:\\{i}_dataset", exist_ok=True)
    os.makedirs(f"D:\\{i}_dataset\\train", exist_ok=True)
    os.makedirs(f"D:\\{i}_dataset\\valid", exist_ok=True)
    os.makedirs(f"D:\\{i}_dataset\\train\\images", exist_ok=True)
    os.makedirs(f"D:\\{i}_dataset\\train\\labels", exist_ok=True)
    os.makedirs(f"D:\\{i}_dataset\\valid\\images", exist_ok=True)
    os.makedirs(f"D:\\{i}_dataset\\valid\\labels", exist_ok=True)

paths = glob.glob(os.path.join('D:\\data001', '*', 'labels', '*.txt'))
paths = paths[:218323]
for path in tqdm(paths):
    tr_vl = path.split('\\')[-3]
    filename = os.path.basename(path)
    labels = []
    yoloxs = []
    yoloys = []
    yolows = []
    yolohs = []
    category = []
    errors = 0
    txt_path = f"D:\\dataset01\\{tr_vl}\\labels\\{filename}"
    with open(txt_path, 'r') as f:
        for i in f.readlines():
            label = i.split(' ')[0]
            yolox = i.split(' ')[1]
            yoloy = i.split(' ')[2]
            yolow = i.split(' ')[3]
            yoloh = i.split(' ')[4]
            # print(label, yolox, yoloy, yolow, yoloh)
            try:
                lb = strd[strd['former_labels'] == int(label)]['labels']
                new_label = int(list(lb)[0])
            except:
                print(txt_path)
                print(label)
                # print(lb)
                exit()

            large_category = strd[strd['former_labels'] == int(label)]['div_l']
            large_category = list(large_category)[0]
            category.append(large_category)

            labels.append(new_label)
            yoloxs.append(yolox)
            yoloys.append(yoloy)
            yolows.append(yolow)
            yolohs.append(yoloh)
    try:
        with open(os.path.join(f'D:\\{lg_category}_dataset', f'{tr_vl}', 'labels', f'{filename}'), 'w') as f:
            f.truncate(0)
    except:
        pass

    lg_category = lg_dict[category[0]]
    for new_label, yolox, yoloy, yolow, yoloh in zip(labels, yoloxs, yoloys, yolows, yolohs):
        with open(os.path.join(f'D:\\{lg_category}_dataset', f'{tr_vl}', 'labels', f'{filename}'), 'a') as f:
            f.write(f'{new_label} {yolox} {yoloy} {yolow} {yoloh} \n')

    shutil.copy(os.path.join("D:\\data001", f"{tr_vl}", 'images', f"{filename.replace('.txt', '.jpg')}"),
                os.path.join(f"D:\\{lg_category}_dataset", f'{tr_vl}', 'images', f"{filename.replace('.txt', '.jpg')}"))
