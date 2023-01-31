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
class WineLabelsDataset(CocoDataset):
    CLASSES = ('wine-labels', 'AlcoholPercentage', "Appellation AOC DOC AVARegion", "Appellation QualityLevel",
               "CountryCountry", "Distinct Logo", "Established YearYear", "Maker-Name", "Organic",
               "Sustainable", "Sweetness-Brut-SecSweetness-Brut-Sec", "TypeWine Type", "VintageYear")

# config
config_file ='.\\configs\\dynamic_rcnn\\dynamic_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)
# print(cfg.pretty_text)

# Learning rate setting
# Single GPU -> 0.0025
# cfg.optimizer.lr = 0.02/8
cfg.optimizer.lr = 0.0025

# dataset setting
cfg.dataset_type = 'WineLabelsDataset'
cfg.data_root = '.\\dataset'

# train, val, test dataset >> type data root ann file img_prefix setting
cfg.data.train.type = 'WineLabelsDataset'
cfg.data.train.ann_file = './dataset/train/_annotations.coco.json'
cfg.data.train.img_prefix = './dataset/train/'

# val
cfg.data.val.type = 'WineLabelsDataset'
cfg.data.val.ann_file = './dataset/valid/_annotations.coco.json'
cfg.data.val.img_prefix = './dataset/valid/'

# test
cfg.data.test.type = 'WineLabelsDataset'
cfg.data.test.ann_file = './dataset/test/_annotations.coco.json'
cfg.data.test.img_prefix = './dataset/test/'

# Class number
cfg.model.roi_head.bbox_head.num_classes = 13

# small obj를 잡기 위해 change anchor -> df: size 8 -> size 4
cfg.model.rpn_head.anchor_generator.scales = [4]

# pretrained call
cfg.load_from = '.\\dynamic_rcnn_r50_fpn_1x-62a3f276.pth'

# train_model save dir
cfg.work_dir = '.\\work_dirs\\0130'

# lr hyp setting
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# cocodataset evaluation type = bbox
# mAP iou threshold 0.5 ~ 0.95
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 10
cfg.checkpoint_config.interval = 10

# epoch setting
# 8 * 12 = 96
cfg.runner.max_epochs = 88
cfg.seed = 777
cfg.data.samples_per_gpu = 6    # single gpu 일 경우 2개는 거의 고정
cfg.data.workers_per_gpu = 2    #
# print('cfg.data >>', cfg.data)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
set_random_seed(777, deterministic=False)
print('cfg info >>', cfg.pretty_text)

datasets = [build_dataset(cfg.data.train)]
print('dataset[0]', datasets[0])

# datasets[0].__dict__ variables key val
datasets[0].__dict__.keys()

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

if __name__ == '__main__' :
    train_detector(model, datasets,cfg,distributed=False, validate=True)