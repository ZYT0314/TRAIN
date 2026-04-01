import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# 设置随机数种子，保证结果可复现
np.random.seed(42)

# 生成300个均匀分布的x值（覆盖双曲线的双象限特征）
x_neg = np.linspace(-0.5, -0.05, 150)  # 第二象限部分（150个点）
x_pos = np.linspace(0.05, 0.5, 150)    # 第四象限部分（150个点）
x_train = np.concatenate([x_neg, x_pos])  # 合并为300个点

# 计算对应的真实y值：使用双曲线函数 y = -0.05 / x（双象限分布）
k = -0.05
y_train = k / x_train  # 核心：替换为双曲线函数

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_train.reshape(-1, 1),
    y_train,
    test_size=0.3,
    random_state=42
)

# 指定模型路径
model_path = "tabpfn-v2-regressor.ckpt"

# 初始化TabPFN回归器
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

# 读取双曲线真实数据文件（由之前的生成代码保存）
input_df = pd.read_excel('hyperbola-True40.xlsx')  # 对应双曲线数据文件
x_test = input_df.iloc[:, 0].values  # 读取第一列x值

# 进行预测
y_pred = reg.predict(x_test.reshape(-1, 1))

# 用预测值替换原DataFrame中的第二列y值
input_df.iloc[:, 1] = y_pred

# 保存预测结果
input_df.to_excel('hyperbola-Tab40.xlsx', index=False)  # 文件名体现双曲线+模型

# 读取真实数据和预测数据
true_data = pd.read_excel('hyperbola-True40.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

pred_data = pd.read_excel('hyperbola-Tab40.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 绘制对比图（不显示y轴刻度）
plt.scatter(x_true, y_true, color='orange', label='True')
plt.scatter(x_pred, y_pred, color='blue', label='Prediction')

plt.title('True vs Predicted', fontsize=20)
plt.xlabel('', fontsize=20)
plt.ylabel('', fontsize=20)
plt.grid(False)

plt.xlim(-0.55, 0.55)
plt.xticks([-0.5, 0, 0.5], fontsize=28)  # 保留x轴刻度
plt.ylim(-0.55, 0.55)
plt.yticks([])  # 清空y轴刻度，不显示任何y轴刻度值

plt.legend(fontsize=18)
plt.tight_layout()
plt.show()