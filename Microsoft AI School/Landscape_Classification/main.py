import os

from custom_dataset import custom_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torch.optim as optim
from utils import train


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 순서가 바뀌면 작동 안할수도 있음
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 순서가 바뀌면 작동 안할수도 있음
])

# train val test Dataset
train_dataset = custom_dataset(".\\dataset\\train", transform=train_transform)
val_dataset = custom_dataset('.\\dataset\\val', transform=val_transform)
test_dataset = custom_dataset('.\\dataset\\test', transform=val_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=126, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=126, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model Call
net = models.resnet18(pretrained=True)
in_feature_val = net.fc.in_features
net.fc = nn.Linear(in_feature_val, 4)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# train model
os.makedirs('.\\model', exist_ok=True)
train(100, train_loader, val_loader, net, optimizer, criterion, device, '.\\model\\best.pt')