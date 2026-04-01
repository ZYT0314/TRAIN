import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置随机数种子，保证结果可复现
np.random.seed(42)

# 生成 300 个均匀分布的 x 值（更适合捕捉周期性特征）
x_train = np.linspace(-0.5, 0.5, 300)
# 计算对应的真实 y 值：使用调整后的余弦函数（自然在[-0.5, 0.5]范围内）
y_train = 0.25 * np.cos(4 * np.pi * x_train) + 0.5 * x_train  # 核心修改：替换为余弦函数

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_train.reshape(-1, 1),
    y_train,
    test_size=0.3,
    random_state=42
)

# 创建并训练 CatBoost 回归模型（参数保持不变）
model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
model.fit(x_train, y_train)

# 进行预测
predictions = model.predict(x_test)

# 计算评估指标
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# 读取余弦函数真实数据文件
input_df = pd.read_excel('Cosx-True50.xlsx')
x_test = input_df.iloc[:, 0].values  # 读取第一列 x 值

# 进行预测
y_pred = model.predict(x_test.reshape(-1, 1))

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred

# 保存结果到新的 Excel 文件
input_df.to_excel('Cosx-Cat50.xlsx', index=False)

# 读取真实数据和预测数据
true_data = pd.read_excel('Cosx-True50.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

pred_data = pd.read_excel('Cosx-Cat50.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 绘制真实数据散点图（橙色）
plt.scatter(x_true, y_true, color='orange', label='True')

# 绘制预测数据散点图（蓝色）
plt.scatter(x_pred, y_pred, color='blue', label='Prediction')

# 保持原图形参数设置不变
plt.title('True vs Predicted', fontsize=20)
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示
plt.grid(False)  # 不显示网格线

plt.xlim(-0.55, 0.55)  # 设置 x 轴范围
plt.xticks([-0.5, 0, 0.5], fontsize=32)  # 设置 x 轴刻度
plt.ylim(-0.55, 0.55)  # 设置 y 轴范围
plt.yticks([-0.5, 0, 0.5], fontsize=32)  # 设置 y 轴刻度

plt.legend(fontsize=22)  # 显示图例
plt.tight_layout()  # 优化布局
plt.show()
