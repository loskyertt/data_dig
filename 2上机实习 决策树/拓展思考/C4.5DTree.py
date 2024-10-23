import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 分裂特征
        self.threshold = threshold  # 分裂阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的类别


class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain_ratio(self, y, X_column, threshold):
        # 计算父节点的熵
        parent_entropy = self._entropy(y)

        # 根据阈值分割数据
        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0

        # 计算子节点的熵
        n = len(y)
        n_l, n_r = len(y[left_mask]), len(y[right_mask])
        e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # 计算信息增益
        info_gain = parent_entropy - child_entropy

        # 计算分裂信息
        split_info = -((n_l/n) * np.log2(n_l/n) + (n_r/n) * np.log2(n_r/n))

        # 返回信息增益率
        return info_gain / split_info if split_info != 0 else 0

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain_ratio(
                    y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 检查停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 寻找最佳分裂点
        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 创建子节点
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx, threshold, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# 训练和评估模型
def evaluate_model(X_train, X_test, y_train, y_test):
    # 创建和训练模型
    model = C45DecisionTree(max_depth=5)
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算准确率
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    # 计算混淆矩阵
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)

    return model, train_accuracy, test_accuracy, conf_matrix_train, conf_matrix_test


if __name__ == "__main__":

    data = pd.read_csv("2上机实习 决策树/拓展思考/environment_data.csv")

    # 准备数据
    X = data.drop('空气等级', axis=1).values
    le = LabelEncoder()
    y = le.fit_transform(data['空气等级'])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练和评估模型
    model, train_accuracy, test_accuracy, conf_matrix_train, conf_matrix_test = evaluate_model(X_train, X_test, y_train, y_test)

    # 打印结果
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 打印混淆矩阵
    print("\n训练集混淆矩阵:")
    print(conf_matrix_train)

    print("\n测试集混淆矩阵:")
    print(conf_matrix_test)

    # 特征重要性分析
    feature_names = ['SO2', 'NO', 'NO2', 'NOx', 'PM10', 'PM2-5']
    importance_dict = {}

    def calculate_feature_importance(node, importance_dict, depth=0):
        if node.value is not None:  # 叶节点
            return

        if node.feature not in importance_dict:
            importance_dict[node.feature] = 0
        importance_dict[node.feature] += 1.0 / (depth + 1)  # 根据深度加权

        calculate_feature_importance(node.left, importance_dict, depth + 1)
        calculate_feature_importance(node.right, importance_dict, depth + 1)

    calculate_feature_importance(model.root, importance_dict)

    # 归一化特征重要性
    total_importance = sum(importance_dict.values())
    normalized_importance = {feature_names[k]: v/total_importance
                            for k, v in importance_dict.items()}

    # 打印特征重要性
    print("\n特征重要性:")
    for feature, importance in sorted(normalized_importance.items(),
                                    key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")


