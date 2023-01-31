import cv2
import json
import os
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.datasets import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed

# Configuration (Train과 동일)

# 1. Dynamic RCNN setting
config_file = './configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)


# 2. Dataset register
@DATASETS.register_module(force=True)
class AnimalDetectionDataset(CocoDataset):
    CLASSES = ('cat', 'chicken', 'cow', 'dog', 'fox', 'goat', 'horse', 'person', 'racoon', 'skunk')


# 3. dataset setting hyp
cfg.dataset_type = 'AnimalDetectionDataset'
cfg.data_root = './dataset'

# train test val dataset >> type, data_root, ann_file, img_prefix setting
cfg.data.train.type = 'AnimalDetectionDataset'
cfg.data.train.ann_file = './dataset/train/_annotations.coco.json'
cfg.data.train.img_prefix = './dataset/train/'

cfg.data.val.type = 'AnimalDetectionDataset'
cfg.data.val.ann_file = './dataset/valid/_annotations.coco.json'
cfg.data.val.img_prefix = './dataset/valid/'

cfg.data.test.type = 'AnimalDetectionDataset'
cfg.data.test.ann_file = './dataset/test/_annotations.coco.json'
cfg.data.test.img_prefix = './dataset/test/'

# 4. model class number setting ...
cfg.model.roi_head.bbox_head.num_classes = 10

# 5. pretrained model
cfg.load_from = './dynamic_rcnn_r50_fpn_1x-62a3f276.pth'

# 6. weight file save dir setting ...
cfg.work_dir = './work_dirs/animal'

# 7. train setting hyp
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# 8. CocoDataset metric -> bbox (bbox mAP iou threshold 0.5 ~ 0.95)
cfg.evaluation.metric = 'mAP'
cfg.evaluation.interval = 10
cfg.checkpoint_config.interval = 10

# Epoch setting
cfg.runner.max_epochs = 10
cfg.seed = 727
cfg.gpu_ids = range(1)
set_random_seed(727, deterministic=False)

# Model call
checkpoint_file = './work_dirs/animal/epoch_10.pth'
model = init_detector(cfg, checkpoint_file, device='cuda')

# one image result show
# from mmdet.apis import show_result_pyplot
# img = './dataset/test/5967_jpg.rf.63927f8cd875d49cedfd362d502cfa05.jpg'
# image = cv2.imread(img)
# results = inference_detector(model, image)
# show_result_pyplot(model, img, results)

img_info_path = './dataset/test/_annotations.coco.json'
with open(img_info_path, 'r', encoding='utf-8') as f:
    image_info = json.loads(f.read())

# threshold
score_threshold = 0.7
submission_anno = list()

for img_info in image_info['images']:
    file_name = img_info['file_name']
    img_height = img_info['height']
    img_width = img_info['width']
    img_path = os.path.join('./dataset/test/', file_name)
    image = cv2.imread(img_path)
    image_copy = image.copy()
    image_resize = cv2.resize(image_copy, (960, 540))


    # scale
    x_scale = float(960 / img_width)
    y_scale = float(540 / img_height)
    # 1280, 720 도 가능

    results = inference_detector(model, img_path)

    for number, result in enumerate(results):
        if len(results) == 0:
            continue

        category_id = number + 1
        # threshold setting
        result_filtered = result[np.where(result[:, 4] > score_threshold)]
        if len(result_filtered) == 0:
            continue
        for i in range(len(result_filtered)):
            # print(result_filtered)
            tmp_dict = dict()
            x_min = result_filtered[i, 0]
            y_min = result_filtered[i, 1]
            x_max = result_filtered[i, 2]
            y_max = result_filtered[i, 3]

            # voc -> coco xywh
            json_x = x_min
            json_y = y_min
            json_w = x_max - x_min
            json_h = y_max - y_min



            tmp_dict['bbox'] = [str(json_x), str(json_y), str(json_w), str(json_h)]
            tmp_dict['category_id'] = category_id
            tmp_dict['area'] = str(json_w * json_h)
            tmp_dict['image_id'] = img_info['id']
            tmp_dict['score'] = float(result_filtered[i, 4])

            submission_anno.append(tmp_dict)


            # scale bbox
            x1 = int(x_min * x_scale)
            y1 = int(y_min * y_scale)
            x2 = int(x_max * x_scale)
            y2 = int(y_max * y_scale)

            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.rectangle(image_resize, (x1, y1), (x2, y2), (0, 255, 255), 2)


    # cv2.imshow('test', image_resize)
    # if cv2.waitKey() == ord('q'):
    #     exit()
    print(submission_anno)
    with open('.\\test2.json', 'w', encoding='utf-8') as f:
        json.dump(submission_anno, f, indent=4, sort_keys=True, ensure_ascii=False)