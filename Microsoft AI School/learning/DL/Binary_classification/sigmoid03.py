# b값의 변화에 따른 좌 우 이동
# b값에 따라서 그래프가 어떻게 변하는지 확인 !
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(1 + x)
y2 = sigmoid(x)
y3 = sigmoid(2 + x)

plt.plot(x, y1, 'r:')  # w = 0.5
plt.plot(x, y2, 'g-')  # w = 1
plt.plot(x, y3, 'b--')  # w = 2
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선
plt.title('Sigmoid Function')
plt.show()

"""
bias에 따른 그래프의 이동
"""