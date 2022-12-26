import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# multi linear
x1_train = torch.FloatTensor([[74], [93], [85], [21], [24]])
x2_train = torch.FloatTensor([[34], [23], [25], [2], [43]])
x3_train = torch.FloatTensor([[7], [13], [65], [71], [34]])
y_train = torch.FloatTensor([[152], [2], [35], [242], [342]])

# 가중치 w 편향 b
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 경사하강법 구현
optimizer = torch.optim.SGD([w1, w2, w3, b], lr=1e-9)

# 학습 진행
epoch_num = 10000

# train loop
for epoch in range(epoch_num + 1):
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    # loss
    loss = torch.mean((hypothesis - y_train) ** 2)
    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # optimizer update

    if epoch % 100 == 0:
        print('epoch {:4d}/{} W1 : {:3f} W2 : {:3f} W3 : {:3f} b : {:.3f} loss : {:.6f}'.format(
            epoch, epoch_num, w1.item(), w2.item(), w3.item(), b.item(), loss
        ))
