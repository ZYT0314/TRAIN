import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置随机数种子，保证结果可复现
np.random.seed(42)

# 生成300个均匀分布的x值（覆盖双曲线的双象限特征）
# 分两段生成：避开x=0，覆盖负区间和正区间
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

# 创建并训练 SVM 回归模型（参数保持不变）
model = SVR(kernel='rbf', C=100, gamma=0.1)
model.fit(x_train, y_train)

# 进行预测
predictions = model.predict(x_test)

# 计算评估指标
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# 读取双曲线真实数据文件（由之前的生成代码保存）
input_df = pd.read_excel('hyperbola-True40.xlsx')  # 对应双曲线数据文件
x_test = input_df.iloc[:, 0].values  # 读取第一列 x 值

# 进行预测
y_pred = model.predict(x_test.reshape(-1, 1))

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred

# 保存结果到新的 Excel 文件
input_df.to_excel('hyperbola-SVM40.xlsx', index=False)  # 文件名体现双曲线+模型

# 读取真实数据和预测数据
true_data = pd.read_excel('hyperbola-True40.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

pred_data = pd.read_excel('hyperbola-SVM40.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 绘制真实数据散点图（橙色）
plt.scatter(x_true, y_true, color='orange', label='True')

# 绘制预测数据散点图（蓝色）
plt.scatter(x_pred, y_pred, color='blue', label='Prediction')

# 保持原图形参数设置不变（不显示y轴刻度）
plt.title('True vs Predicted', fontsize=20)
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示
plt.grid(False)  # 不显示网格线

plt.xlim(-0.55, 0.55)  # 设置 x 轴范围
plt.xticks([-0.5, 0, 0.5], fontsize=28)  # 设置 x 轴刻度
plt.ylim(-0.55, 0.55)  # 设置 y 轴范围
plt.yticks([])  # 不显示y轴刻度

plt.legend(fontsize=18)  # 显示图例
plt.tight_layout()  # 优化布局
plt.show()