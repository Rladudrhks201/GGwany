import sys

import albumentations as A
import torchvision.models
from albumentations.pytorch import ToTensorV2
from custom_dataset import custom_dataset
from utils import visualize_augmentations, test
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import pandas as pd
import os
import glob

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09, rotate_limit=25, p=0.1),
        A.Resize(width=224, height=224),
        A.RandomBrightnessContrast(p=0.6),
        A.RandomFog(),
        A.RandomShadow(p=0.6),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = custom_dataset('.\\dataset\\train', transform=train_transform)
    val_dataset = custom_dataset('.\\dataset\\valid', transform=val_transform)
    test_dataset = custom_dataset('.\\dataset\\test', transform=val_transform)

    # visualize_augmentations(train_dataset, idx=20, samples=10)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    # cpu - gpu 로 올릴때 pin_memory 사용

    # MODEL CALL

    # Swin_b
    # net = models.swin_b(weights='IMAGENET1K_V1')
    # net.head = nn.Linear(in_features=1024, out_features=450, bias=True)
    # net.to(device)
    # print(net)

    # Resnet50
    # net = models.resnet50(pretrained=True)
    # net.fc = nn.Linear(in_features=2048, out_features=450)
    # net.to(device)

    # Mobilenet_v3_small
    net = models.mobilenet_v3_small(pretrained=True)
    net.classifier[3] = nn.Linear(in_features=1024, out_features=450, bias=True)
    net.to(device)

    num_epochs = 10
    criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.0005)

    best_val_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    os.makedirs('.\\models', exist_ok=True)
    save_path = '.\\models\\best.pt'
    dfForAccuracy = pd.DataFrame(index=list(range(num_epochs)), columns=['Epoch', 'Accuracy'])
    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv('.\\ModelAccuracy.csv')['Accuracy'].tolist())

    for epoch in range(num_epochs):
        running_loss = 0
        val_acc = 0
        train_acc = 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = f'Train Epoch [{epoch + 1}/{num_epochs}], loss >> {loss.data:.3f}'

        net.eval()
        with torch.no_grad():
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='green')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)

                val_acc += (torch.argmax(outputs, 1) == labels).sum().item()

        val_accuracy = val_acc / len(test_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
        print(f'Epoch [{epoch + 1}/{num_epochs}]',
              f'Train Loss : {(running_loss / train_steps):.3f}',
              f'Train Acc : {train_accuracy:.3f}\t Val Acc : {val_accuracy:.3f}')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch == num_epochs - 1:
            dfForAccuracy.to_csv('.\\ModelAccuracy.csv', index=False)

if __name__ == '__main__':
    # main()

    # test
    test()



