from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 차원 축소의 다른 방법 LDA , PCA와 차이점은 선형관계를 최대한 유지하려함

iris = datasets.load_iris()
features = iris.data
target = iris.target

lda = LinearDiscriminantAnalysis(n_components= 1)
features_lda = lda.fit(features, target).transform(features)

print("원본 특성의 개수 : ", features.shape[1])
print("줄어든 특성의 개수 : ", features_lda.shape[1])

# 설명된 분산의 비율이 담긴 배열을 저장
lda_var_ratios = lda.explained_variance_ratio_
print(lda_var_ratios)

def select_n_components(var_ratio, goal_val : float) -> int:
    total_variances = 0.0 # 설명된 분산의 초기값을 지정
    n_components = 0 # 특성 개수의 초기값을 지정

    for experienced_variance in var_ratio:
        total_variances += experienced_variance
        n_components += 1 # 성분 개수를 카운트
        if total_variances >= goal_val:
            break
    return n_components

temp = select_n_components(lda_var_ratios, 0.95)
print("temp = ",temp)
# 95% 이상의 설명력을 갖추려면 필요한 최소한의 변수 개수