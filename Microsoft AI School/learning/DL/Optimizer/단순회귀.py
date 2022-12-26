import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear model

# 랜덤 시드 설정
torch.manual_seed(7272)

# 실습을 위한 기본세팅 훈련데이터, x_train, y_train
x_train = torch.FloatTensor(([1], [2], [3]))
y_train = torch.FloatTensor(([2], [4], [6]))
# print(x_train, x_train.shape)
# print(y_train, y_train.shape)
"""
tensor([[1.],
        [2.],
        [3.]]) torch.Size([3, 1])
tensor([[2.],
        [4.],
        [6.]]) torch.Size([3, 1])
"""

# 가중치와 편향의 초기화
# requires_grad = True -> 학습을 통해 계속 값이 변경되는 변수
w = torch.zeros(1, requires_grad=True)
# print(w)
b = torch.zeros(1, requires_grad=True)

# 가설
# 직선의 방정식
hypothesis = x_train * w + b

# declare loss function
# MSE
loss = torch.mean((hypothesis - y_train) ** 2)
# print(loss)

# 경사하강법 구현
optimizer = torch.optim.SGD([w, b], lr=0.01)

# 기울기 0으로 초기화
optimizer.zero_grad()
loss.backward()

# 학습 진행
epoch_num = 2000

# train loop
for epoch in range(epoch_num + 1):
    hypothesis = x_train * w + b
    # loss
    loss = torch.mean((hypothesis - y_train) ** 2)
    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # optimizer update

    if epoch % 100 == 0:
        print('epoch {:4d}/{} W : {:3f} b : {:.3f} loss : {:.6f}'.format(
            epoch, epoch_num, w.item(), b.item(), loss
        ))