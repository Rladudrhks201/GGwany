import cv2
import json
import glob
import os
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.datasets import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed


def voc_json_to_coco_json(pathlist, tr_vl):
    label_dict = {0: 'garbage_bag', 1: 'sit_board', 2: 'street_vendor', 3: 'food_truck', 4: 'banner', 5: 'tent',
                  6: 'smoke',
                  7: 'flame', 8: 'pet', 9: 'bench', 10: 'park_pot', 11: 'trash_can', 12: 'rest_area', 13: 'toilet',
                  14: 'street_lamp', 15: 'park_info'}
    label_dict = {v: k for k, v in label_dict.items()}
    classes = []
    # {"id": 10, "name": "skunk", "supercategory": "animals"}
    for k, v in label_dict.items():
        classes.append({'id': v, 'name': k})
    image_no = 0
    box_id = 0
    image_list = list()
    submission_anno = list()

    for path in pathlist:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                image_info = json.loads(f.read())
        except:
            print(path)
            exit()

        img_info = image_info['images']
        file_name = img_info['ori_file_name']
        if file_name[0] == '0':
            file_name = file_name[1:]
        img_height = int(img_info['height'])
        img_width = int(img_info['width'])
        img_path = os.path.join(f'A:/test01/{tr_vl}/images', file_name)

        # scale
        x_scale = float(960 / img_width)
        y_scale = float(540 / img_height)

        results = image_info['annotations']
        for number, result in enumerate(results):
            if len(results) == 0:
                continue

            category_name = result['object_class']
            try:
                category_id = label_dict[category_name]
            except:
                # print(category_name)
                continue
            bbox = result['bbox']

            tmp_dict = dict()
            x_min = float(bbox[0][0])
            y_min = float(bbox[0][1])
            x_max = float(bbox[1][0])
            y_max = float(bbox[1][1])
            # print(bbox)
            # print(x_min, y_min, x_max, y_max)

            # voc -> coco xywh
            json_x = round(x_min * x_scale, 6)
            json_y = round(y_min * y_scale, 6)
            json_w = round((x_max - x_min) * x_scale, 6)
            json_h = round((y_max - y_min) * y_scale, 6)

            tmp_dict['id'] = int(box_id)
            tmp_dict['image_id'] = int(image_no)
            tmp_dict['category_id'] = int(category_id)
            tmp_dict['bbox'] = [int(json_x), int(json_y), int(json_w), int(json_h)]
            tmp_dict['area'] = int(round(json_w * json_h, 6))
            tmp_dict['segmentation'] = []
            tmp_dict['iscrowd'] = 0

            submission_anno.append(tmp_dict)
            box_id += 1

        if len(submission_anno) == 0:
            os.remove(path)
            os.remove(img_path)
            continue
        i_info = {'id': int(image_no), 'license': 1, 'file_name': f'{file_name}', 'height': int(img_height / 2),
                  'width': int(img_width / 2)}
        image_list.append(i_info)

        image_no += 1

    json1 = {}
    json1['categories'] = classes
    json1['images'] = image_list
    json1['annotations'] = submission_anno

    if tr_vl == 'Train':
        pth = '.\\train.json'
    elif tr_vl == 'Valid':
        pth = '.\\valid.json'
    else:
        pth = '.\\test.json'

    with open(pth, 'w', encoding='utf-8') as f:
        json.dump(json1, f, indent=4, sort_keys=False, ensure_ascii=False)


if __name__ == '__main__':
    train_json_path = glob.glob(os.path.join('D:\\dataset2\\dataset', 'Train', 'labels', '*.json'))
    val_json_path = glob.glob(os.path.join('D:\\dataset2\\dataset', 'Valid', 'labels', '*.json'))
    test_json_path = glob.glob(os.path.join('D:\\dataset2\\dataset', 'Test', 'labels', '*.json'))
    # print(len(train_json_path) + len(val_json_path))
    # print(train_json_path)
    voc_json_to_coco_json(train_json_path, 'Train')
    voc_json_to_coco_json(val_json_path, 'Valid')
    voc_json_to_coco_json(test_json_path, 'Test')
