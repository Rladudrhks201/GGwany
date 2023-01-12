import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
from custom_dataset import custom_dataset
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import pandas as pd
import os
import glob
import warnings

warnings.filterwarnings(action='ignore')

def train_val():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = custom_dataset('.\\dataset\\train', transform=train_transform)
    val_dataset = custom_dataset('.\\dataset\\valid', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # net = models.resnet50(pretrained=True)
    # net.fc = nn.Linear(in_features=2048, out_features=11)
    # net.to(device)

    # net = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
    # net.fc = nn.Linear(in_features=1024, out_features=11, bias=True)
    # net.to(device)

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(in_features=512, out_features=11)
    net.to(device)

    num_epochs = 20
    criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.005)

    best_val_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    os.makedirs('.\\models', exist_ok=True)
    save_path = '.\\models\\best.pt'
    dfForAccuracy = pd.DataFrame(index=list(range(num_epochs)), columns=['Epoch', 'train_loss', 'train_acc',
                                                                         'val_loss', 'val_acc'])

    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv('.\\ModelAccuracy.csv')['val_acc'].tolist())

    for epoch in range(num_epochs):
        running_loss = 0
        val_losses = 0
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
                loss = criterion(outputs, labels)
                val_losses += loss.item()

                val_acc += (torch.argmax(outputs, 1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'train_loss'] = round(running_loss / train_steps, 3)
        dfForAccuracy.loc[epoch, 'train_acc'] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, 'val_loss'] = round(val_losses / val_steps, 3)
        dfForAccuracy.loc[epoch, 'val_acc'] = round(val_accuracy, 3)

        print(f'Epoch [{epoch + 1}/{num_epochs}]',
              f'Train Loss : {(running_loss / train_steps):.3f}',
              f'Train Acc : {train_accuracy:.3f},'
              f'Val Loss : {(val_losses / val_steps):.3f}',
              f' Val Acc : {val_accuracy:.3f}')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch % 5 == 0:
            torch.save(net.state_dict(), '.\\models\\last.pt')

        if epoch % 5 == 0:
            dfForAccuracy.to_csv('.\\ModelAccuracy.csv', index=False)

def test():
    test_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(),
        ToTensorV2()
    ])
    test_dataset = custom_dataset('.\\dataset\\test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # resnet50
    # net = models.resnet50(pretrained=False)
    # net.fc = nn.Linear(in_features=2048, out_features=11)
    # net.to(device)

    # net = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
    # net.fc = nn.Linear(in_features=1024, out_features=11, bias=True)
    # net.to(device)

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(in_features=512, out_features=11)
    net.to(device)

    model_path = '.\\models\\best_shufflenetv2_x1_0.pt'
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    test_loss = 0
    test_steps = len(test_loader)
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout, colour='blue')
        for data in test_bar:
            image, label = data
            images, labels = image.to(device), label.to(device)
            output = net(images)
            test_loss += criterion(output, labels).item()
            _, argmax = torch.max(output, 1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()

        acc = correct / total * 100
        loss = test_loss / test_steps
        print(f'Test Loss >> {loss}', f'Test acc >> {acc}%')

if __name__ == '__main__':
    # 학습
    train_val()


    # 테스트
    # test()

