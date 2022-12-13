import os
import json
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# json 파일 읽기

json_path = 'C:/Users/user/Documents/github/Microsoft AI School/learning/image01/annotations/instances_default.json'
# json_path = 'C:/Users/user/Documents/github/Microsoft AI School/learning/image_kiwi/annotations/instances_default.json'
with open(json_path, 'r') as f:
    coco_info = json.load(f)

# print(coco_info)

assert len(coco_info) > 0, '파일 읽기 실패'

# 카테고리 정보 수집
categories = dict()
for category in coco_info['categories']:
    # print(category)
    categories[category['id']] = category['name']

print('categories info : ', categories)

# annotation 정보 수집
ann_info = dict()
for annotation in coco_info['annotations']:
    # print(annotation)
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    category_id = annotation['category_id']
    # print(f'image_id : {image_id}, category_id : {category_id}, bbox : {bbox}')

    if image_id not in ann_info:
        ann_info[image_id] = {
            'boxes': [bbox], 'categories': [category_id]
        }
    else:
        ann_info[image_id]['boxes'].append(bbox)
        ann_info[image_id]['categories'].append(categories[category_id])

print('annotation : ', ann_info)
ex0 = np.zeros((1, 5))
csv0 = pd.DataFrame(ex0, columns=['file_name', 'x1', 'y1', 'w', 'h'])

# xml 생성
tree = ET.ElementTree()
root = ET.Element('annotations')

for i, image_info in enumerate(coco_info['images']):
    # xml file save folder
    os.makedirs('C:\\Users\\user\\Desktop\\project\\xml_folder\\', exist_ok=True)
    xml_save_path = 'C:\\Users\\user\\Desktop\\project\\xml_folder\\test.xml'

    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']
    # print(filename, width, height, img_id)
    xml_frame = ET.SubElement(root, 'image', id=str(i), name=filename, width='%d' % width,
                              height='%d' % height)

    file_path = os.path.join('C:/Users/user/Documents/github/Microsoft AI School/learning/image01/images',
                             filename)
    img = cv2.imread(file_path)
    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue
    # print(annotation)

    for bbox, category in zip(annotation['boxes'], annotation['categories']):
        x1, y1, w, h = bbox
        # print(x1, y1, w, h)

        # label이 하나가 아닌경우
        # ex) labels_list = {1: 'kiwi', 2:'apple'} , label 자리에 넣는다
        ET.SubElement(xml_frame, 'box', label='kiwi', occluded='0', source='manual', x1='%.3f' % x1,
                      y1='%.3f' % y1, w='%.3f' % w, h='%.3f' % h, z_order='0')

        rec_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (225, 0, 255), 2)

    # cv2.imshow('test', rec_img)
    # cv2.waitKey(0)
    cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\box\\{filename}', rec_img)
    # 한글 경로 때문에 안돼서 다른 곳에 저장

    filename1 = np.zeros((len(annotation['boxes']), 1)).reshape((-1, 1))
    filename1 = pd.DataFrame(filename1, columns=['file_name'])
    filename1['file_name'] = filename.replace('.jpg', '')
    csv1 = pd.DataFrame(annotation['boxes'], columns=['x1', 'y1', 'w', 'h'])
    csv2 = pd.concat([filename1, csv1], axis=1)
    csv0 = pd.concat([csv0, csv2], axis=0)

    # xml 저장
    tree._setroot(root)
    tree.write(xml_save_path, encoding='utf-8')

# csv 저장
csv0 = csv0.reset_index()
csv0 = csv0.iloc[1:, 1:]
print(csv0)
csv0.to_csv('C:/Users/user/Desktop/1213과제_김영관/image.csv')


