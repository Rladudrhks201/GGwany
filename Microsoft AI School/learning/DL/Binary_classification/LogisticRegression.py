# pytorch의 nn.Linear, nn.Sigmoid로 로지스틱 회귀를 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Train_Data -> Tensor
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs_num = 1000

for epoch in range(epochs_num + 1):
    output = model(x_train)
    loss = F.binary_cross_entropy(output, y_train)

    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = output >= torch.FloatTensor([0.5])  # 0.5를 기준으로 True, False 설정
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True
        acc = correct_prediction.sum() / len(correct_prediction)
        print(f'Epoch : {epoch} / {epochs_num} loss : {loss.item():.6f} acc : {acc * 100:.2f}')

print(model(x_train))
