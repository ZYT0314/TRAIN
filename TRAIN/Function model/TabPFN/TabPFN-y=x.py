import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# 设置随机数种子，保证结果可复现
np.random.seed(42)

# 生成 300 个随机的 x 值
x_train = np.random.uniform(-0.5, 0.5, 300)
# 计算对应的真实 y 值，使用函数 y = |x|
y_train = np.abs(x_train)

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

# Predict a point estimate (using the mean)
predictions = reg.predict(x_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# Predict quantiles
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    x_test,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))

# 读取包含 x 和 y 值的数据表格
input_df = pd.read_excel('abs_x21.xlsx')
x_test = input_df.iloc[:, 0].values  # 读取第一列 x 值

# 进行预测
y_pred = reg.predict(x_test.reshape(-1, 1))

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred

# 保存结果到新的 Excel 文件
input_df.to_excel('abs_x21-Tab.xlsx', index=False)

# 读取真实数据
true_data = pd.read_excel('abs_x21.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

# 读取预测数据
pred_data = pd.read_excel('abs_x21-Tab.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 绘制真实数据曲线图（橙色）
plt.scatter(x_true, y_true, color='orange', label='True')

# 绘制预测数据曲线图（蓝色）
plt.scatter(x_pred, y_pred, color='blue', label='Prediction')

# 调整字体大小参数
plt.title('True vs Predicted', fontsize=20)  # 标题字体从14调整为20
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示
plt.grid(False)  # 不显示网格线

plt.xlim(-0.55, 0.55)  # 设置 x 轴范围
plt.xticks([-0.5, 0, 0.5], fontsize=28)  # x轴刻度字体从20调整为28
plt.ylim(-0.55, 0.55)  # 设置 y 轴范围
plt.yticks([-0.5, 0, 0.5], fontsize=28)  # y轴刻度字体从20调整为28

plt.legend(fontsize=18)  # 图例字体设置为18
plt.tight_layout()  # 优化布局
plt.show()