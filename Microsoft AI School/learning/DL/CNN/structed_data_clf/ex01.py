import torch
import pandas as pd
import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class customDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.data = self.df.iloc[:, :3].values
        self.label = self.df.iloc[:, 3].values
        # self.label[self.label == 'True'] = 1
        # self.label[self.label == 'False'] = 0
        self.length = len(self.df)

    def __getitem__(self, index):
        tempdata = self.data[index]
        templabel = self.label[index]
        tempdata = torch.FloatTensor(tempdata)
        templabel = torch.FloatTensor([templabel])
        # print(tempdata)
        # print(templabel)
        return tempdata, templabel

    def __len__(self):
        return self.length


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


train_dataset = customDataset('.\\dataset02.csv')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10001):
    loss_ = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ += loss

    loss_ = loss_ / len(train_loader)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch : {epoch + 1:4d}, loss : {loss:.4f}')

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [[89, 92, 75], [75, 64, 50], [38, 58, 63], [33, 42, 39], [23, 15, 32]]
    ).to(device)
    output = model(inputs)

    print("-------------------------")
    print(output)
    print(output >= torch.FloatTensor([0.5]).to(device))