import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader  # Dataset인데 Tensor로 결과가 나옴

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Tensor Dataset 입력으로 사용하고 dataset 지정합니다
Dataset = TensorDataset(x_train, y_train)

# DataLoader
dataloader = DataLoader(Dataset, batch_size=2, shuffle=True)

# model 설계
model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

epochs_num = 2000
for epoch in range(epochs_num + 1):
    for batch_idx, sample in enumerate(dataloader):
        x_train, y_train = sample
        prediction = model(x_train)
        # loss
        loss = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('Epoch : {:4d}/{} loss : {:.6f}'.format(epoch, epochs_num, loss.item()))