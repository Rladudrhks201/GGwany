import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# print(x_data)
print(x_data.shape)

data2 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]]
x_data2 = torch.tensor(data2)
# print(x_data2)
print(x_data2.shape)  # 차원, 행, 열

# 텐서로 부터 텐서 생성
x_ones = torch.ones_like(x_data)
# print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(x_rand)

shape = (2, 4)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor, '\n', ones_tensor, '\n', zeros_tensor)

tensor = torch.rand(3, 4)
print(tensor.shape, '\n', tensor.dtype, '\n', tensor.device)
# 형태, 데이터 타입, 연결된 기기 (이 경우에는 cpu)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
else:
    tensor = tensor.to('cpu')

print(tensor.device)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1.shape)
print(tensor * tensor)
print(tensor @ tensor.T)  # 3x4 @ 4x3

print(tensor.add_(10))  # 모든 원소에 10을 더함

# 텐서 numpy 배열 변환
t = torch.ones(5)
n = t.numpy()
print(type(n), '\n', n)

# numpy 배열 텐서 변환
n = np.ones(5)
n = np.add(n, 1)
t = torch.from_numpy(n)
print(type(t), '\n', t)

# view
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

print(ft.view([-1, 3]).shape)  # 한차원을 낮춤
print(ft.view([-1, 1, 3]).shape)

# squeeze
ft = torch.FloatTensor([[1], [2], [3]])
print(torch.squeeze(ft).shape)

# unsqueeze
ft = torch.FloatTensor([1, 2, 3])
print(ft.unsqueeze(0).shape)    # 인자는 행과 열 가중치의 옵션

#
