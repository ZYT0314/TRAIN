import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# 创建 XGBoost 数据集
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

# 设置参数
params = {
    'objective':'reg:squarederror',
    'eval_metric': ['rmse','mae'],
   'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# 训练模型
num_round = 100
model = xgb.train(params, dtrain, num_round)

# 进行预测
predictions = model.predict(dtest)

# 计算评估指标
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# 读取包含 x 和 y 值的数据表格
input_df = pd.read_excel('abs_x21.xlsx')
x_test = input_df.iloc[:, 0].values  # 读取第一列 x 值

# 创建 XGBoost 格式的测试数据
dtest_excel = xgb.DMatrix(x_test.reshape(-1, 1))

# 进行预测
y_pred = model.predict(dtest_excel)

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred

# 保存结果到新的 Excel 文件
input_df.to_excel('abs_x21-XGB.xlsx', index=False)

# 读取真实数据
true_data = pd.read_excel('abs_x21.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

# 读取预测数据
pred_data = pd.read_excel('abs_x21-XGB.xlsx')
x_pred = pred_data.iloc[:, 0].values
y_pred = pred_data.iloc[:, 1].values

# 绘制真实数据散点图（橙色）
plt.scatter(x_true, y_true, color='orange', label='True')

# 绘制预测数据散点图（蓝色）
plt.scatter(x_pred, y_pred, color='blue', label='Prediction')

plt.title('True vs Predicted', fontsize=20)
plt.xlabel('', fontsize=20)  # 取消x轴字母显示
plt.ylabel('', fontsize=20)  # 取消y轴字母显示
plt.grid(False)  # 不显示网格线

plt.xlim(-0.55, 0.55)  # 设置 x 轴范围
plt.xticks([-0.5, 0, 0.5], fontsize=28)  # 设置 x 轴刻度
plt.ylim(-0.55, 0.55)  # 设置 y 轴范围
plt.yticks([-0.5, 0, 0.5], fontsize=28)  # 设置 y 轴刻度

plt.legend(fontsize=18)  # 显示图例
plt.tight_layout()  # 优化布局
plt.show()