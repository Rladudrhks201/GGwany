# 다중 선형 회귀 클래스 선언
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# class 생성
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear.forward(x)


# model 정의
model = MultivariateLinearRegressionModel()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5)

# train
epochs_num = 2000
for epoch in range(epochs_num + 1):
    prediction = model(x_train)
    # loss
    loss = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch : {:4d}/{} loss : {:.6f}'.format(epoch, epochs_num, loss.item()))

new_var = torch.FloatTensor([[73, 82, 72]])
pred_y = model(new_var)
print(f'예측 값은 {pred_y.item()}')
