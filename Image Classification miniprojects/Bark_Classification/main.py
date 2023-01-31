import os

from custom_dataset import Custom_Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import rexnetv1
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.Resize(height=224, width=224),
    A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    A.RandomSnow(p=0.3),
    A.GaussianBlur(p=0.25),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.Resize(height=224, width=224),
    A.Normalize(),
    ToTensorV2()
])

# Dataset
train_dataset = Custom_Dataset('.\\dataset\\train', transform=train_transform)
val_dataset = Custom_Dataset('.\\dataset\\val', transform=val_transform)
test_dataset = Custom_Dataset('.\\dataset\\test', transform=val_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=126, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=126, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=126, shuffle=False, num_workers=3)

# pretrain start
# model = rexnetv1.ReXNetV1()
# model.load_state_dict(torch.load('.\\rexnetv1_1.0.pth'))
# model.output[1] = nn.Conv2d(1280, 50, kernel_size=1, stride=1)
# model.to(device)

# no pretrain start
model = rexnetv1.ReXNetV1(classes=50)
model.load_state_dict(torch.load('.\\models\\best.pt'))  # 모델 기반으로 돌림
model.to(device)

criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
os.makedirs('.\\models', exist_ok=True)
save_dir = '.\\models'
num_epochs = 100

if __name__ == '__main__':
    # train(num_epochs, model, train_loader, val_loader, criterion, optimizer, save_dir, device)
    test(model, test_loader, device)
