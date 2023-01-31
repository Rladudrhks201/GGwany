import glob
import os
import json
import cv2
import shutil
from sklearn.model_selection import train_test_split


# folder create
os.makedirs('.\\dataset\\train', exist_ok=True)
os.makedirs('.\\dataset\\train\\images', exist_ok=True)
os.makedirs('.\\dataset\\train\\labels', exist_ok=True)
os.makedirs('.\\dataset\\test', exist_ok=True)
os.makedirs('.\\dataset\\test\\images', exist_ok=True)
os.makedirs('.\\dataset\\test\\labels', exist_ok=True)
os.makedirs('.\\dataset\\valid', exist_ok=True)
os.makedirs('.\\dataset\\valid\\images', exist_ok=True)
os.makedirs('.\\dataset\\valid\\labels', exist_ok=True)
os.makedirs('.\\dataset\\temp', exist_ok=True)

# json path
train_json_path = '.\\dataset\\train\\_annotations.coco.json'
test_json_path = '.\\dataset\\test\\_annotations.coco.json'
val_json_path = '.\\dataset\\valid\\_annotations.coco.json'
json_path_list = [train_json_path, test_json_path, val_json_path]


# VOC -> YOLO format
def json_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

for pth in json_path_list:
    if pth == train_json_path:
        split = 'train'
    elif pth == test_json_path:
        split = 'test'
    else:
        split = 'valid'
    # json open
    with open(pth, 'r') as f:
        info = json.load(f)
    label_dict = {}
    for category in info['categories']:
        label_dict[category['name']] = category['id']
    ann = info['annotations']
    for i, ann_info in enumerate(ann):
        bbox = ann_info['bbox']
        label_number = ann_info['category_id']
        id = ann_info['id']
        image_id = ann_info['image_id']

        image_info = info['images']
        image_name = image_info[image_id]['file_name']
        image_name_ = image_info[image_id]['file_name'][:-4]
        h = image_info[image_id]['height']
        w = image_info[image_id]['width']

        bbox = json_to_yolo_bbox(bbox, w, h)
        yolo_x = str(round(bbox[0], 6))
        yolo_y = str(round(bbox[1], 6))
        yolo_w = str(round(bbox[2], 6))
        yolo_h = str(round(bbox[3], 6))

        with open(f'.\\dataset\\{split}\\labels\\{image_name_ + ".txt"}', 'a') as f:
                f.write(f'{label_number} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n')
        if os.path.exists(f'.\\dataset\\{split}\\labels\\{image_name_ + ".txt"}') == False:
            with open(f'.\\dataset\\{split}\\labels\\{image_name_ + ".txt"}', 'w') as f:
                f.write('')

img_path = glob.glob(os.path.join('.\\dataset', '*', '*.jpg'))
for path in img_path:
    if 'test' in path:
        split = 'test'
    elif 'train' in path:
        split = 'train'
    else:
        split = 'valid'
    name = os.path.basename(path)
    shutil.move(path, f'.\\dataset\\{split}\\images\\{name}')