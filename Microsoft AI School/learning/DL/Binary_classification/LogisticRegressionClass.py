# 클래스로 파이토치 로지스틱 회귀 모델 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Train_Data -> Tensor
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model.forward(x)


# call model
model = BinaryClassifier()
# print(model)
"""
BinaryClassifier(
  (model): Sequential(
    (0): Linear(in_features=2, out_features=1, bias=True)
    (1): Sigmoid()
  )
)
"""

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# epoch
epochs_num = 1000

# train
for epoch in range(epochs_num):
    output = model(x_train)
    loss = F.binary_cross_entropy(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = output >= torch.FloatTensor([0.5])  # 0.5를 기준으로 True, False 설정
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True
        acc = correct_prediction.sum() / len(correct_prediction)
        print(f'Epoch : {epoch} / {epochs_num} loss : {loss.item():.6f} acc : {acc * 100:.2f}')

print(model(x_train))
