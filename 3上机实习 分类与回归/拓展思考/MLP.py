import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import  r2_score
import matplotlib.pyplot as plt

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载和预处理
def load_and_preprocess_data():

    data = pd.read_csv('3上机实习 分类与回归/demo/tmp/new_reg_data_GM11.csv')

    # 分离训练数据和预测数据
    train_data = data[data['y'].notna()].copy()  # 1994-2013的数据
    future_data = data[data['y'].isna()].copy()  # 2014-2015的数据

    # 特征列
    feature_cols = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']

    # 分离特征和目标变量
    X_train = train_data[feature_cols]
    y_train = train_data['y']
    X_future = future_data[feature_cols]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_future_scaled = scaler_X.transform(X_future)

    return X_scaled, y_scaled, X_future_scaled, scaler_X, scaler_y

# 2. 创建和训练MLP模型
def train_mlp_model(X, y):
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    model.fit(X, y.ravel())
    return model

# 3. 模型评估
def evaluate_model(model, X, y):
    y_pred = model.predict(X).reshape(-1, 1)
    r2 = r2_score(y, y_pred)
    return y_pred, r2

# 4. 预测未来值
def predict_future(model, X_future_scaled, scaler_y):
    future_pred_scaled = model.predict(X_future_scaled).reshape(-1, 1)
    future_predictions = scaler_y.inverse_transform(future_pred_scaled)
    return future_predictions

# 5. 可视化结果
def plot_results(actual, predicted, future_predictions):
    plt.figure(figsize=(12, 6))

    # 训练数据（1994-2013）
    train_years = range(1994, 2014)
    # 预测数据（2014-2015）
    future_years = [2014, 2015]

    # 绘制实际值和预测值
    plt.plot(train_years, actual, 'b-', label='实际值')
    plt.plot(train_years, predicted, 'r--', label='预测值')
    plt.plot(future_years, future_predictions, 'g--', label='未来预测')

    plt.title('财政收入预测分析')
    plt.xlabel('年份')
    plt.ylabel('财政收入')
    plt.legend()
    plt.grid(True)

    return plt

# 主程序执行
def main():
    try:
        # 加载和预处理数据
        X_scaled, y_scaled, X_future_scaled, scaler_X, scaler_y = load_and_preprocess_data()

        # 训练模型
        model = train_mlp_model(X_scaled, y_scaled)

        # 评估模型
        y_pred_scaled, r2 = evaluate_model(model, X_scaled, y_scaled)

        # 转换回原始比例
        y_actual = scaler_y.inverse_transform(y_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # 预测未来值
        future_predictions = predict_future(model, X_future_scaled, scaler_y)

        # 打印评估指标和预测结果
        print(f"决定系数 (R²): {r2:.6f}")
        print("\n2014年预测值:", future_predictions[0][0])
        print("2015年预测值:", future_predictions[1][0])

        # 可视化结果
        plt = plot_results(y_actual, y_pred, future_predictions)
        plt.show()

    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()