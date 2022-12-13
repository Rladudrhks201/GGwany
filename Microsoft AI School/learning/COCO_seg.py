import os
import json
import cv2
import numpy as np

# json 파일 읽기

json_path = 'C:/Users/user/Documents/github/Microsoft AI School/learning/image01/annotations/instances_seg.json'

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
    segmentation = annotation['segmentation']

    # print(f'image_id : {image_id}, category_id : {category_id}, bbox : {bbox}, segmentation : {segmentation}')

    if image_id not in ann_info:
        ann_info[image_id] = {
            'boxes': [bbox], 'segmentation': [segmentation], 'categories': [category_id]
        }
    else:
        ann_info[image_id]['boxes'].append(bbox)
        ann_info[image_id]['segmentation'].append(segmentation)
        ann_info[image_id]['categories'].append(categories[category_id])

# print('annotation : ', ann_info)

for image_info in coco_info['images']:
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']
    # print(filename, width, height, img_id)
    file_path = os.path.join('C:/Users/user/Documents/github/Microsoft AI School/learning/image01/images',
                             filename)
    img = cv2.imread(file_path)
    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue
    # print(annotation)

    for bbox, segmentation, category in zip(annotation['boxes'],
                                            annotation['segmentation'], annotation['categories']):
        x1, y1, w, h = bbox
        # print(x1, y1, w, h)
        for seg in segmentation:
            poly = np.array(seg, np.int32).reshape((int(len(seg) / 2), 2))  # 2열짜리 행렬로 변환하여 한 쌍씩 출력
            poly_img = cv2.polylines(img, [poly], True, (255, 0, 0), 2)
    cv2.imwrite(f'C:\\Users\\user\\Desktop\\project\\poly\\{filename}', poly_img)
    cv2.imshow('test', poly_img)
    cv2.waitKey(0)



