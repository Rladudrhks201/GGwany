import numpy as np
import matplotlib.pyplot as plt

# [-10,10] 구간에서 100개의 t값을 시그모이드 함수에 대입
t = np.linspace(-10, 10, 100)

# sigmoid
sig = 1 / (1 + np.exp(-t))

# t와 시그모이드 결과 그래프
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], 'k-')
plt.plot([-10, 10], [0.5, 0.5], 'k:')
plt.plot([-10, 10], [1, 1], 'k:')
plt.plot([0, 0], [-1.1, 1.1], 'k-')
plt.plot(t, sig, 'r-', linewidth=2, label=r'$\sigma(t) = \frac{1}{1 + e^{-t}}$')
plt.xlabel('t')
plt.legend(loc='upper left', fontsize=25)
plt.axis([-10, 10, -0.1, 1.1])  # 그래프 간격
plt.show()
