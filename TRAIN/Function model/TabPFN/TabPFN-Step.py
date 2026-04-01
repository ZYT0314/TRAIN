import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# 设置随机数种子，保证结果可复现
np.random.seed(42)

# 定义区间边界
x_boundaries = np.linspace(-0.5, 0.5, 6)
# 每个阶梯对应60个数据点
points_per_step = 60
total_points = len(x_boundaries) * points_per_step

# 生成均匀分布的x值
x = np.zeros(total_points)
y = np.zeros(total_points)
for i in range(len(x_boundaries) - 1):
    start_idx = i * points_per_step
    end_idx = (i + 1) * points_per_step
    # 计算每个小区间的步长
    step = (x_boundaries[i + 1] - x_boundaries[i]) / (points_per_step + 1)
    # 生成当前区间内均匀分布的x值
    x[start_idx:end_idx] = [x_boundaries[i] + (j + 1) * step for j in range(points_per_step)]
    # 计算当前区间对应的y值
    y[start_idx:end_idx] = [-0.4 + i * 0.2] * points_per_step

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x.reshape(-1, 1),
    y,
    test_size=0.3,
    random_state=42
)

# 指定模型路径
model_path = "tabpfn-v2-regressor.ckpt"

# 初始化TabPFN回归器
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

# 读取包含x和y值的数据表格
input_df = pd.read_excel('step_function25.xlsx')
x_test = input_df.iloc[:, 0].values  # 读取第一列x值

# 进行预测
y_pred = reg.predict(x_test.reshape(-1, 1))

# 用预测值替换原DataFrame中的第二列y值
input_df.iloc[:, 1] = y_pred

# 保存结果到新的Excel文件
input_df.to_excel('step_function25-Tab.xlsx', index=False)

# 读取真实数据
true_data = pd.read_excel('step_function25.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

# 读取预测数据
pred_data = pd.read_excel('step_function25-Tab.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 处理真实数据，按阶梯分段
true_segments_x = []
true_segments_y = []
start_index = 0
for i in range(1, len(x_true)):
    if y_true[i] != y_true[i - 1]:
        true_segments_x.append(x_true[start_index:i])
        true_segments_y.append([y_true[start_index]] * (i - start_index))
        start_index = i
true_segments_x.append(x_true[start_index:])
true_segments_y.append([y_true[start_index]] * (len(x_true) - start_index))

# 定义区间边界
x_boundaries = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]

# 绘制真实数据曲线图（橙色），线宽3
for x_seg, y_seg in zip(true_segments_x, true_segments_y):
    plt.plot(x_seg, y_seg, color='orange', label='True', linewidth=3)

# 绘制预测数据曲线（蓝色），线宽1.5
for i in range(len(x_boundaries) - 1):
    start_bound = x_boundaries[i]
    end_bound = x_boundaries[i + 1]
    mask = (x_pred >= start_bound) & (x_pred < end_bound)
    x_seg = x_pred[mask]
    y_seg = y_pred[mask]
    if len(x_seg) > 0:
        plt.plot(x_seg, y_seg, color='blue', label='Prediction' if i == 0 else "", linewidth=1.5)

# **按指定格式修改图形参数**
plt.title('True vs Predicted', fontsize=20)  # 标题字体从14调整为20
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示
plt.grid(False)  # 不显示网格线

plt.xlim(-0.55, 0.55)  # x轴范围
plt.xticks([-0.5, 0, 0.5], fontsize=28)  # x轴刻度字体从20调整为28
plt.ylim(-0.55, 0.55)  # y轴范围
plt.yticks([-0.5, 0, 0.5], fontsize=28)  # y轴刻度字体从20调整为28

# 图例字体大小设置为28（与坐标轴刻度一致）
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=18)

plt.tight_layout()  # 优化布局
plt.show()