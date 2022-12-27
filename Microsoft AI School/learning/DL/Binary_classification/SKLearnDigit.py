# 다중 퍼셉트론으로 손글씨 분류
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
import torch.nn as nn

digits = load_digits()
# print(digits.keys())
"""
dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
"""
# print(digits.target_names)
"""
[0 1 2 3 4 5 6 7 8 9]   # 총 10개
"""

image_and_label_list = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(image_and_label_list[:4]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample : %i' % label)
plt.show()

# train data, label
x = digits.data
y = digits.target

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())  # learning rate -> default

losses = []  # loss graph
epochs_num = 100

for epoch in range(epochs_num + 1):
    output = model(x)
    loss = loss_fun(output, y.long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch : [{:4d}/{}] loss : {:.6f}'.format(
            epoch, epochs_num, loss.item()
        ))

    losses.append(loss.item())

plt.title("loss")
plt.plot(losses)
plt.show()
