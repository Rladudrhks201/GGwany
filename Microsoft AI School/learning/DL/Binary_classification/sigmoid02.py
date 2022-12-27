# w값의 변화에 따른 경사도 변화
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)

plt.plot(x, y1, 'r:')  # w = 0.5
plt.plot(x, y2, 'g-')  # w = 1
plt.plot(x, y3, 'b--')  # w = 2
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선
plt.title('Sigmoid Function')
plt.show()

"""
W 값이 커질 수록, 경사도가 커진다
"""
