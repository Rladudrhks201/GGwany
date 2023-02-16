import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm


def xml_to_L_label_csv():
    # xml, jpg path
    xml_path = glob.glob(os.path.join('D:\\resize_dataset\\dataset', '*', 'labels', '*', '*_meta.xml'))
    transform_dict = {}

    for path in tqdm(xml_path):
        tr_vl = path.split('\\')[-4]
        folder_name = path.split('\\')[-2]

        tree = ET.parse(path)
        root = tree.getroot()

        element = root.find('div_cd')
        item_no = element.find('item_no').text
        if item_no not in transform_dict.keys():
            div_l = element.find('div_l').text
            div_m = element.find('div_m').text
            div_s = element.find('div_s').text
            div_n = element.find('div_n').text
            temp_dict = {}
            temp_dict['div_l'] = div_l
            temp_dict['div_m'] = div_m
            temp_dict['div_s'] = div_s
            temp_dict['div_n'] = div_n
            transform_dict[item_no] = temp_dict

    lb_df = pd.DataFrame.from_dict(transform_dict, orient='index')
    lb_df.to_csv('.\\Large_label_standard.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    xml_to_L_label_csv()
