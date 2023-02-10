#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import pandas as pd
from tqdm import tqdm


# VOC -> YOLO format
def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def xml_to_yolo():
    # label dict
    label_df = pd.read_csv('.\\final.csv', encoding='utf-8')
    label_df.columns = ['code', 'name']
    code, name = label_df['code'], label_df['name']
    index = label_df.index
    label_dict = {}
    for index, name in zip(index, name):
        name = name.replace(u'\xa0', u'')
        label_dict[name] = int(index)
    label_dict['오뚜기황태콩나물해장국밥301.5G'] = 3317
    # print(label_dict)

    # label standard csv
    label_standard = pd.read_csv('.\\label_standard.csv', encoding='utf-8')


    # xml, jpg path
    xml_path = glob.glob(os.path.join('D:\\dataset01', '*', 'labels', '*', '*_meta.xml'))


    for path in tqdm(xml_path):
        tr_vl = path.split('\\')[-4]
        folder_name = path.split('\\')[-2]

        w = 2988
        h = 2988
        tree = ET.parse(path)
        root = tree.getroot()

        element = root.find('annotation')
        file_name = element.find('filename').text
        file_name = file_name.replace('.jpg', '.txt')
        for member in element.findall('object'):
            label = member.find('name').text
            try:
                if int(folder_name) in list(label_standard['folder']):
                    label = label_standard[label_standard['folder'] == int(folder_name)]['new']
                        # label = str(list(label)[0])
                    label = str(label.values)
                    label = label.strip(']').strip('[').strip("'")
                label = label_dict[label]
            except:
                label = label_dict[label]

            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            bbox = [xmin, ymin, xmax, ymax]
            bbox = xml_to_yolo_bbox(bbox, w, h)
            yolo_x = str(round(bbox[0], 3))
            yolo_y = str(round(bbox[1], 3))
            yolo_w = str(round(bbox[2], 3))
            yolo_h = str(round(bbox[3], 3))

            os.makedirs(f'D:\\tempp\\{tr_vl}', exist_ok=True)
            os.makedirs(f'D:\\tempp\\{tr_vl}\\labels', exist_ok=True)
            os.makedirs(f'D:\\tempp\\{tr_vl}\\labels\\{folder_name}', exist_ok=True)

            with open(f'D:\\tempp\\{tr_vl}\\labels\\{folder_name}\\{file_name}', 'a') as f:
                f.write(f'{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n')
        # if os.path.exists(f'D:\\tempp\\{tr_vl}\\labels\\{folder_name}\\{file_name}') == False:
        #     with open(f'D:\\tempp\\{tr_vl}\\labels\\{folder_name}\\{file_name}', 'w') as f:
        #         f.write('')


if __name__ == '__main__':
    xml_to_yolo()
