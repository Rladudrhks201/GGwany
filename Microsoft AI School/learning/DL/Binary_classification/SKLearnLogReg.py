import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# iris data
iris = datasets.load_iris()
list_iris = iris.keys()
print(list_iris)
print(iris.feature_names)
x = iris['data'][:, 3:]  # 꽃잎의 너비 변수 활용
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
y = (iris['target'] == 2).astype('int')  # ris-versinica인 경우에만 1

log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(x, y)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = log_reg.predict_proba(x_new)

# 시각화 1
# # 음성 클래스
# plt.plot(x_new, y_prob[:, 0], 'g-', label='Iris-Virginica')
# # 양성
# plt.plot(x_new, y_prob[:, 1], 'r--', label='Not Iris-Virginica')
# plt.legend()
# plt.show()

# 시각화 2
plt.figure(figsize=(8, 3))
decision_boundary = x_new[y_prob[:, 1] >= 0.5][0]   # 확률이 0.5 넘게나오게 만든 최소 꽃잎의 너비
plt.plot(x[y == 0], y[y == 0], 'bs')  # 음성 범주
plt.plot(x[y == 1], y[y == 1], 'g^')  # 양성 범주
# 결정경계 표시
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)

plt.plot(x_new, y_prob[:, 0], 'g-', label='Iris-Virginica')
plt.plot(x_new, y_prob[:, 1], 'r--', label='Not Iris-Virginica')

# 결정 결계 표시
plt.text(decision_boundary + 0.02, 0.15, 'Decision boundary', fontsize=14, color='k', ha='center')
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel('petal width(cm)', fontsize=14)
plt.ylabel('probability', fontsize=14)
plt.legend(loc='center left', fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()

"""
결정 경계가 어떤 값을 가지고 분류하는가?
양쪽의 확률이 0.5가 되는 1.6cm 근방에서 결정 경계가 만들어지고 분류기는 경계를 기준으로 결과를 예측
"""

test_data = log_reg.predict([[1.8], [1.48]])
print(test_data)    # 예상대로 1 0의 결과가 나옴!
