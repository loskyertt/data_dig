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
        self.left = left
        self.right = right
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

    # 信息增益率计算
    def _information_gain_ratio(self, y, X_column, threshold):
        # 父节点的熵
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


# 模型预测
def predict_air_quality(model, le, input_data):
    """
    预测空气质量等级

    Parameters:
    -----------
    model: C45DecisionTree
        训练好的C4.5决策树模型
    le: LabelEncoder
        标签编码器
    input_data: dict or pandas.DataFrame
        需要预测的数据，包含SO2, NO, NO2, NOx, PM10, PM2-5的值

    Returns:
    --------
    str: 预测的空气质量等级
    dict: 包含详细信息的字典
    """
    # 转换输入数据为DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    # 确保列的顺序正确
    required_columns = ['SO2', 'NO', 'NO2', 'NOx', 'PM10', 'PM2-5']
    input_df = input_df[required_columns]

    # 转换为numpy数组进行预测
    X = input_df.values

    # 进行预测
    prediction_encoded = model.predict(X)

    # 转换预测结果
    prediction = le.inverse_transform(prediction_encoded)

    # 如果是单个样本，返回详细信息
    if len(prediction) == 1:
        result = {
            'predicted_class': prediction[0],
            'input_values': input_data,
            'message': f"预测的空气质量等级为: {prediction[0]}"
        }
        return prediction[0], result

    # 如果是多个样本，返回所有预测结果
    return prediction, {'predicted_classes': prediction.tolist()}


if __name__ == "__main__":

    data = pd.read_csv("2上机实习 决策树/拓展思考/environment_data.csv")

    X = data.drop('空气等级', axis=1).values
    # 转换为独热编码
    le = LabelEncoder()
    y = le.fit_transform(data['空气等级'])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    print()

    # 预测单样本
    new_sample = {
        'SO2': 0.020,
        'NO': 0.001,
        'NO2': 0.045,
        'NOx': 0.046,
        'PM10': 0.070,
        'PM2-5': 0.045
    }
    predicted_class, details = predict_air_quality(model, le, new_sample)
    print(details['message'])

    # 预测多个样本
    new_samples = pd.DataFrame([
        {'SO2': 0.020, 'NO': 0.001, 'NO2': 0.045, 'NOx': 0.046, 'PM10': 0.070, 'PM2-5': 0.045},
        {'SO2': 0.025, 'NO': 0.002, 'NO2': 0.055, 'NOx': 0.057, 'PM10': 0.080, 'PM2-5': 0.055}
    ])
    predictions, _ = predict_air_quality(model, le, new_samples)
    print("\n多个样本预测结果:")
    for i, pred in enumerate(predictions):
        print(f"样本 {i+1} 预测的空气质量等级: {pred}")