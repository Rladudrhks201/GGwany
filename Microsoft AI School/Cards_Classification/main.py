import argparse
import os

import torch
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from custom_dataset import custom_dataset
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from utils import *
from adamp import AdamP


# pip install adamp

def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.8),
        A.HorizontalFlip(p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomShadow(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # dataset
    train_dataset = custom_dataset(file_path=opt.train_path, transform=train_transform)
    val_dataset = custom_dataset(file_path=opt.val_path, transform=val_transform)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    # model call
    net = models.__dict__['resnet50'](pretrained=True)
    net.fc = nn.Linear(512, 53)
    net.to(device)

    # loss
    criterion = LabelSmoothingCrossEntropy().to(device)

    # optimizer
    optimizer = AdamP(net.parameters(), lr=opt.lr, weight_decay=1e-2)

    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
    # 60, 90마다 떨어짐, 0.1 만큼

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # 스텝 사이즈마다 떨어짐, 0.1 만큼

    # model, pt save dir
    save_dir = opt.save_path
    os.makedirs(save_dir, exist_ok=True)

    # train, val
    # train(num_epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, device)
    train(10, net, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, device)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default='.\\train',
                        help='train data path')
    parser.add_argument("--val-path", type=str, default='.\\valid',
                        help='val data path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='lr number')
    parser.add_argument('--save-path', type=str, default='.\\weights',
                        help='save path')
    opt = parser.parse_args()

    return opt


# visualize_aug(train_dataset)    # Visualize

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
