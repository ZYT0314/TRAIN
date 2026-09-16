import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机数种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)

# ==============================================================================
# 1. TRAIN 模型核心：自监督掩码预训练模块 (Self-Supervised Pre-training)
# ==============================================================================
print("===== TRAIN 模型自监督预训练阶段 =====")

# 生成 1000 个均匀分布的 x 值及其对应的三次函数 y 值（作为预训练知识库）
x_pretrain = np.linspace(-0.5, 0.5, 1000).reshape(-1, 1)
y_pretrain = 4 * x_pretrain ** 3  # 依据新的三次函数规律

# 定义特征编码器和解码器
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, dropout=0.1):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)  # 提取到高维特征空间
        )
    def forward(self, x):
        return self.encoder(x)

class FeatureDecoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super(FeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

encoder = FeatureEncoder()
decoder = FeatureDecoder()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.MSELoss()

# 准备预训练 DataLoader
pretrain_dataset = TensorDataset(torch.FloatTensor(x_pretrain))
pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

epochs = 30
mask_prob = 0.2

for epoch in range(epochs):
    for batch in pretrain_loader:
        x_batch = batch[0]
        # TRAIN 模型掩码策略
        mask = torch.rand(x_batch.shape) < mask_prob
        x_masked = x_batch.clone()
        x_masked[mask] = 0.0  # 掩盖部分特征
        
        optimizer.zero_grad()
        encoded = encoder(x_masked)
        decoded = decoder(encoded)
        
        # 仅对未被掩盖或被掩盖的部分进行重构学习
        valid_mask = ~mask
        if valid_mask.sum() > 0:
            loss = criterion(decoded[valid_mask], x_batch[valid_mask])
            loss.backward()
            optimizer.step()

print("预训练完成，TRAIN 特征编码器已就绪。\n")

# 定义一个辅助函数：用预训练好的编码器转换数据
def extract_features(x_np):
    encoder.eval()
    with torch.no_grad():
        return encoder(torch.FloatTensor(x_np)).numpy()


# ==============================================================================
# 2. 原有逻辑：生成 300 个样本并进行建模
# ==============================================================================
print("===== 下游回归拟合与评估阶段 =====")

# 生成 300 个均匀分布的 x 值（适合捕捉三次函数特征）
x_train_raw = np.linspace(-0.5, 0.5, 300)  # 均匀分布覆盖整个定义域，更贴合三次函数特征
# 计算对应的真实 y 值：使用三次函数 y = 4x³（自然在[-0.5, 0.5]范围内）
y_train = 4 * x_train_raw ** 3  # 核心：替换为三次函数

# 划分训练集和测试集
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x_train_raw.reshape(-1, 1),
    y_train,
    test_size=0.3,
    random_state=42
)

# 【核心修改】将原始数据通过 TRAIN 编码器提取特征
x_train_enc = extract_features(x_train_raw)
x_test_enc = extract_features(x_test_raw)

# 指定模型路径
model_path = "tabpfn-v2-regressor.ckpt"

# 初始化 TabPFN 回归器
reg = TabPFNRegressor(model_path=model_path)
# 使用提取后的特征进行拟合
reg.fit(x_train_enc, y_train)

# 预测并计算评估指标
predictions = reg.predict(x_test_enc)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# 分位数预测
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    x_test_enc,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))


# ==============================================================================
# 3. 外部数据预测与 Excel 处理
# ==============================================================================
# 读取三次函数真实数据文件（由之前的生成代码保存）
input_df = pd.read_excel('ax3.xlsx')  # 对应三次函数数据文件
x_ext_raw = input_df.iloc[:, 0].values.reshape(-1, 1)  # 读取第一列 x 值

# 【核心修改】外部数据也必须经过 TRAIN 编码器
x_ext_enc = extract_features(x_ext_raw)

# 进行预测
y_pred_ext = reg.predict(x_ext_enc)

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred_ext

# 保存预测结果
input_df.to_excel('ax3-TRAIN50.xlsx', index=False)  # 文件名体现三次函数+模型


# ==============================================================================
# 4. 原有逻辑：绘图
# ==============================================================================
# 读取真实数据和预测数据
true_data = pd.read_excel('ax3.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

pred_data = pd.read_excel('ax3-TRAIN50.xlsx')
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