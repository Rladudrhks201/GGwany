import torch
import torch.nn as nn
import torch.nn.functional as F


# 이미지 크기 -> (H + 2 * P - FH) / S + 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)  # (20 - 5) / 1 + 1 = 16
        self.fc1 = nn.Linear(10 * 12 * 12, 50)  # (16 - 5) / 1  + 1  = 12
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        print('연산전 x.size >>', x.size())
        x = F.relu(self.conv1(x))
        print('연산1 후 x.size >>', x.size())
        x = F.relu(self.conv2(x))
        print('연산2 후 x.size >>', x.size())
        x = x.view(-1, 10 * 12 * 12)  # reshape와의 차이는 reshape는 원본을 복사 후에 사이즈 변경이라 원본은 변경 X
                                      # 또 view의 값이 변경되면 원본도 변경되지만 reshape의 값이 변경되도 원본은 변경 X
        print('차원 감소 후 >>', x.size())
        x = F.relu(self.fc1(x))
        print('fc1 연산 후 x.size >>', x.size())
        x = F.relu(self.fc2(x))
        print('fc2 연산 후 x.size >>', x.size())
        """
        연산전 x.size >> torch.Size([10, 1, 20, 20])
        연산1 후 x.size >> torch.Size([10, 3, 16, 16])
        연산2 후 x.size >> torch.Size([10, 10, 12, 12])
        차원 감소 후 >> torch.Size([10, 1440])
        fc1 연산 후 x.size >> torch.Size([10, 50])
        fc2 연산 후 x.size >> torch.Size([10, 10])
        """
        return x


cnn = CNN()
output = cnn(torch.randn(10, 1, 20, 20))
