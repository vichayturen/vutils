
import math

import numpy as np


class ReliefFeatureSelection:
    """
    过滤式特征选择方法，注意输入的X中离散变量以独热乘\sqrt{2}/2的形式，连续变量使用MinMax归一化
    """
    def __init__(self, select_ratio: float = 0.5):
        self.select_ratio = select_ratio
        self.weights = None
        self.selected_features = None

    def fit(self, X, y, **kwargs):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for i in range(n_samples):
            # 找到同类和不同类的最近邻
            same_class_neighbors = np.where(y == y[i])[0]
            diff_class_neighbors = np.where(y != y[i])[0]

            # 计算距离
            distances_same = np.linalg.norm(X[i] - X[same_class_neighbors], axis=1)
            distances_diff = np.linalg.norm(X[i] - X[diff_class_neighbors], axis=1)

            # 找到最近邻
            nearest_same = np.argmin(distances_same)
            nearest_diff = np.argmin(distances_diff)

            # 更新权重
            self.weights -= np.pow(X[i] - X[same_class_neighbors][nearest_same], 2).flatten()
            self.weights += np.pow(X[i] - X[diff_class_neighbors][nearest_diff], 2).flatten()

        # 根据阈值筛选特征
        rank_weights = np.argsort(self.weights)
        remains_feature_num = math.ceil(n_features * self.select_ratio)
        self.selected_features = np.zeros(n_features, dtype=bool)
        self.selected_features[rank_weights[-remains_feature_num:]] = True

    def transform(self, X):
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    scalar = MinMaxScaler()
    X = np.array([[1, 2, 3, 4], [9, 10, 3, 4], [3, 4, 11, 12], [13, 14, 15, 16]])
    X_normalized = scalar.fit_transform(X)
    y = np.array([0, 1, 0, 1])

    relief = ReliefFeatureSelection(0.5)
    relief.fit(X_normalized, y)
    X_new = relief.transform(X)
    print(X_new)
    print(relief.weights)
