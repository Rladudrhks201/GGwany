from DataSet import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
# from hy_parameter import *
import hy_parameter
from torchvision import models
import torch.nn as nn
from utils import train, validate, save_model

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train aug
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])
# val aug
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# dataset
train_dataset = custom_dataset('C:\\Users\\user\\Desktop\\Search\\Data\\train', transform=train_transform)
val_dataset = custom_dataset('C:\\Users\\user\\Desktop\\Search\\Data\\val', transform=val_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=hy_parameter.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hy_parameter.batch_size, shuffle=False)

# model call
# net = models.__dict__['resnet18'](pretrained=False, num_classes=hy_parameter.num_classes)
net = models.__dict__['resnet18'](pretrained=True)
# pretrained model , num_classes 4로 수정 방법
net.fc = nn.Linear(512, 4)
net.to(device)

# criterion
# criterion = nn.BCEWithLogitsLoss().to(device)   # BCE는 binary라서 사용 불가
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=hy_parameter.lr)

# save dir
model_save_dir = './model/'

# train, val, eval, save
train(number_epoch=hy_parameter.epoch, train_loader=train_loader, val_loader=val_loader,
      criterion=criterion, optimizer=optim, model=net, save_dir=model_save_dir, device=device)


