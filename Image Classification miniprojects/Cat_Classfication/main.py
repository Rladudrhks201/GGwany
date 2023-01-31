import os
import sys
import warnings
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import aug_function
from custom_dataset import custom_dataset
import torchvision
import torch.nn as nn
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

warnings.filterwarnings(action='ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = aug_function(mode_flag='train')
val_transform = aug_function(mode_flag='val')

train_dataset = custom_dataset('.\\dataset\\train', transform=train_transform)
val_dataset = custom_dataset('.\\dataset\\val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=3)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# Model Call
efficient_net = models.efficientnet_b3(pretrained=True)
# print(efficient_net)  # Layer확인
efficient_net.classifier[1] = nn.Linear(1536, 7)
efficient_net.to(device)



if __name__ == '__main__':
    criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(efficient_net.parameters(), lr=0.0003, weight_decay=0.0005)
    num_epochs = 50
    os.makedirs('.\\models', exist_ok=True)
    save_path = '.\\models\\best.pt'
    best_val_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    dfForAccuracy = pd.DataFrame(index=list(range(num_epochs)),
                                 columns=['Epoch', 'Accuracy'])

    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv('.\\models\\modelAccuracy.csv')['Accuracy'].tolist())
        efficient_net.load_state_dict(torch.load(save_path))

    for epoch in range(num_epochs):
        running_loss = 0
        val_acc = 0
        train_acc = 0

        efficient_net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='red')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = efficient_net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f'train epoch [{epoch + 1} / {num_epochs + 1}], loss {loss.data:.3f}'

        efficient_net.eval()
        with torch.no_grad():
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = efficient_net(images)
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
        print(f'Epoch [{epoch + 1}/{num_epochs}]',
              f'Train Loss : {(running_loss / train_steps):.3f}',
              f'Train Acc : {train_accuracy:.3f}\t Val Acc : {val_accuracy:.3f}')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(efficient_net.state_dict(), save_path)

        if epoch == num_epochs - 1:
            dfForAccuracy.to_csv('.\\modelAccuracy.csv', index=False)
