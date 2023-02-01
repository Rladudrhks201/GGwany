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


def voc_json_to_coco_json(pathlist):
    label_dict = {1: 'garbage_bag', 2: 'sit_board', 3: 'street_vendor', 4: 'food_truck', 5: 'banner', 6: 'tent',
                  7: 'smoke',
                  8: 'flame', 9: 'pet', 10: 'bench', 11: 'park_pot', 12: 'trash_can', 13: 'rest_area', 14: 'toilet',
                  15: 'street_lamp', 16: 'park_info'}
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
        tr_vl = path.split('\\')[-3]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                image_info = json.loads(f.read())
        except:
            print(path)
            exit()


        img_info = image_info['images']
        file_name = img_info['ori_file_name']
        img_height = int(img_info['height']) * 0.5
        img_width = int(img_info['width']) * 0.5
        img_path = os.path.join('./dataset/test/', file_name)

        # scale
        x_scale = float(960 / img_width)
        y_scale = float(540 / img_height)

        i_info = {'id': f'{image_no}', 'license': 1, 'file_name': f'{file_name}', 'height': f'{img_height}',
                  'width': f'{img_width}'}
        image_list.append(i_info)

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
            json_x = float(round(x_min * x_scale, 6))
            json_y = float(round(y_min * y_scale, 6))
            json_w = float(round((x_max - x_min) * x_scale, 6))
            json_h = float(round((y_max - y_min) * y_scale, 6))

            tmp_dict['id'] = box_id
            tmp_dict['image_id'] = image_no
            tmp_dict['category_id'] = category_id
            tmp_dict['bbox'] = [str(json_x), str(json_y), str(json_w), str(json_h)]
            tmp_dict['area'] = str(round(json_w * json_h, 6))
            tmp_dict['segmentation'] = []
            tmp_dict['iscrowd'] = 0


            submission_anno.append(tmp_dict)
            box_id += 1





        image_no += 1

    json1 = {}
    json1['categories'] = classes
    json1['images'] = image_list
    json1['annotations'] = submission_anno


    if tr_vl == 'Train':
        pth = '.\\train.json'
    elif tr_vl == 'Valid':
        pth = '.\\Valid.json'
    with open(pth, 'w', encoding='utf-8') as f:
        json.dump(json1, f, indent=4, sort_keys=False, ensure_ascii=False)



if __name__ == '__main__':
    train_json_path = glob.glob(os.path.join('A:\\test01', 'Train', 'labels', '*.json'))
    val_json_path = glob.glob(os.path.join('A:\\test01', 'Valid', 'labels', '*.json'))
    # print(len(train_json_path) + len(val_json_path))
    # print(train_json_path)
    voc_json_to_coco_json(train_json_path)
    # voc_json_to_coco_json(val_json_path)
