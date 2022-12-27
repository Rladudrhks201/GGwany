# 파이토치로 다층 퍼셉트론 구현
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# XOR 구현을 위한 입 출력 데이터
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# 참고
# CrossEntropy 경우에는 마지막 레이어 노드수가 2개 이상이여야 한다.
# 만약 마지막 층에 1개 output 이라면 BCELoss를 사용할때가 있다.
# BCELoss 함수를 사용할 경우에는 먼저 마지막 레이어의 값이 0~1로 조정을 해줘야한다.
# 따라서 BCELoss 함수를 쓸땐 마지막 레이어를 시그모이드 함수를 적용시켜야함 !!

"""
제일먼저 CrossEntropy같은경우에는 마지막 레이어 노드수가 2개 이상이여야 한다.
1개일 경우에는 사용이 안됨.

마지막 레이어가 노드수가 1개일 경우에는 보통 Binary Classification을 할때 사용될수가 있는데
이럴경우 BCELoss를 사용할때가 있다.

BCELoss함수를 사용할 경우에는 먼저 마지막 레이어의 값이 0~1로 조정을 해줘야하기 때문에
단순하게 마지막 레이어를 nn.Linear로 할때 out of range 에러가 뜬다.

따라서 BCELoss함수를 쓸땐 마지막 레이어를 시그모이드함수를 적용시켜줘야 한다.
"""

# model
model = nn.Sequential(
    nn.Linear(2, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),
    nn.Sigmoid()    # 사용할 Loss BCELoss 이므로 시그모이드 활성함수 적용
).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs_num = 60000
for epoch in range(epochs_num + 1):
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch : {epoch} loss : {loss.item()}')


# inference code
with torch.no_grad():
    output = model(x)
    predicted = (output > 0.5).float()
    acc = (predicted == y).float().mean()
    print('모델의 출력값 output', output.detach().cpu().numpy())
    print('모델의 예측값 output', predicted.detach().cpu().numpy())
    print('실제 값 Y', y.cpu().numpy())
    print('정확도 -> ', '\n', acc.item())
