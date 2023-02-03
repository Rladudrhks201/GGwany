import mmcv
import torch

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes
from mmcv.runner import get_dist_info, init_dist
from mmcv import Config


# Dataset register
@DATASETS.register_module(force=True)
class ParkDataset(CocoDataset):
    CLASSES = ('garbage_bag', 'sit_board', 'street_vendor', 'food_truck',
               'banner', 'tent', 'smoke', 'flame', 'pet', 'bench',
               'park_pot', 'trash_can', 'rest_area', 'toilet', 'street_lamp', 'park_info')


# config
config_file = '.\\configs\\dynamic_rcnn\\dynamic_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)
# print(cfg.pretty_text)

# Learning rate setting
# Single GPU -> 0.0025
# cfg.optimizer.lr = 0.02/8
cfg.optimizer.lr = 0.0025

# dataset setting
cfg.dataset_type = 'ParkDataset'
cfg.data_root = 'A:\\dataset'

# train, val, test dataset >> type data root ann file img_prefix setting
cfg.data.train.type = 'ParkDataset'
cfg.data.train.ann_file = 'A:/dataset/train/_annotations.coco.json'
cfg.data.train.img_prefix = 'A:/dataset/train/'

# val
cfg.data.val.type = 'ParkDataset'
cfg.data.val.ann_file = 'A:/dataset/valid/_annotations.coco.json'
cfg.data.val.img_prefix = 'A:/dataset/valid/'

# test
cfg.data.test.type = 'ParkDataset'
cfg.data.test.ann_file = 'A:/dataset/test/_annotations.coco.json'
cfg.data.test.img_prefix = 'A:/dataset/test/'

# Class number
cfg.model.roi_head.bbox_head.num_classes = 16

# small obj를 잡기 위해 change anchor -> df: size 8 -> size 4
cfg.model.rpn_head.anchor_generator.scales = [4]

# pretrained call
cfg.load_from = '.\\dynamic_rcnn_r50_fpn_1x-62a3f276.pth'

# train_model save dir
cfg.work_dir = '.\\work_dirs\\park'

# lr hyp setting
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# cocodataset evaluation type = bbox
# all AP iou threshold 0.5 ~ 0.95 precision
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 10
cfg.checkpoint_config.interval = 1

# epoch setting
# 8 * 12 = 96
cfg.runner.max_epochs = 10
cfg.seed = 7277
cfg.data.samples_per_gpu = 6  # single gpu 일 경우 2개는 거의 고정
cfg.data.workers_per_gpu = 2  #
# print('cfg.data >>', cfg.data)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
set_random_seed(7277, deterministic=False)
print('cfg info >>', cfg.pretty_text)

datasets = [build_dataset(cfg.data.train)]
print('dataset[0]', datasets[0])

# datasets[0].__dict__ variables key val
datasets[0].__dict__.keys()

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

if __name__ == '__main__':
    train_detector(model, datasets, cfg, distributed=False, validate=True)
