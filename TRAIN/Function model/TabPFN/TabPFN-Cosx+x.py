import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# 设置随机数种子，保证结果可复现
np.random.seed(42)

# 生成 300 个均匀分布的 x 值（替换随机采样，更适合捕捉周期性特征）
x_train = np.linspace(-0.5, 0.5, 300)  # 均匀分布更贴合余弦函数周期性
# 计算对应的真实 y 值：使用调整后的余弦函数（自然在[-0.5, 0.5]范围内）
y_train = 0.25 * np.cos(4 * np.pi * x_train) + 0.5 * x_train  # 核心：替换为余弦函数

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_train.reshape(-1, 1),
    y_train,
    test_size=0.3,
    random_state=42
)

# 指定模型路径
model_path = "tabpfn-v2-regressor.ckpt"

# 初始化 TabPFN 回归器
reg = TabPFNRegressor(model_path=model_path)
reg.fit(x_train, y_train)

# 预测并计算评估指标
predictions = reg.predict(x_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# 分位数预测
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    x_test,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))

# 读取余弦函数真实数据文件（由之前的生成代码保存）
input_df = pd.read_excel('Cosx-True50.xlsx')
x_test = input_df.iloc[:, 0].values  # 读取第一列 x 值

# 进行预测
y_pred = reg.predict(x_test.reshape(-1, 1))

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred

# 保存预测结果
input_df.to_excel('Cosx-Tab50.xlsx', index=False)

# 读取真实数据和预测数据
true_data = pd.read_excel('Cosx-True50.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

pred_data = pd.read_excel('Cosx-Tab50.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 绘制对比图（保持原格式）
plt.scatter(x_true, y_true, color='orange', label='True')
plt.scatter(x_pred, y_pred, color='blue', label='Prediction')

plt.title('True vs Predicted', fontsize=20)
plt.xlabel('', fontsize=20)
plt.ylabel('', fontsize=20)
plt.grid(False)

plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)
plt.ylim(-0.55, 0.55)
plt.yticks([-0.5, 0, 0.5], fontsize=28)

plt.legend(fontsize=18)
plt.tight_layout()
plt.show()