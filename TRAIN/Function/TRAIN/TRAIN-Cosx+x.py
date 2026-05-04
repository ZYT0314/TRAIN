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
# 1. 增强版 TRAIN 模型：多维数学扩展 + 自监督掩码预训练 (PoC 验证)
# ==============================================================================
print("===== TRAIN 模型自监督预训练阶段 =====")

# 将 1D 扩展为 5D 通用数学基底，模拟高维空间的互相影响
def expand_features(x_array):
    """将单维 x 扩展为 [x, x^2, x^3, sin(x), cos(x)] 5维通用特征空间"""
    x = x_array.flatten()
    expanded = np.vstack([
        x, 
        x**2, 
        x**3, 
        np.sin(x * np.pi), 
        np.cos(x * np.pi)
    ]).T
    return expanded

# 生成 1000 个均匀分布的预训练 x 值
x_pretrain_raw = np.linspace(-0.5, 0.5, 1000).reshape(-1, 1)
# 扩展为 5 维特征矩阵
X_pretrain_expanded = expand_features(x_pretrain_raw)
input_dim = X_pretrain_expanded.shape[1]

# 完全复用 TRAIN 模型的编码器与解码器架构
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, dropout=0.1):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)  # 提取为 8维 高级潜变量
        )
    def forward(self, x):
        return self.encoder(x)

class FeatureDecoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super(FeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

encoder = FeatureEncoder(input_dim=input_dim)
decoder = FeatureDecoder(input_dim=input_dim)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.MSELoss()

# 准备预训练 DataLoader
pretrain_dataset = TensorDataset(torch.FloatTensor(X_pretrain_expanded))
pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

epochs = 30
mask_prob = 0.2  # 20%的概率掩盖特征

for epoch in range(epochs):
    for batch in pretrain_loader:
        x_batch = batch[0]
        
        # 自监督掩码策略 (Masking)
        mask = torch.rand(x_batch.shape) < mask_prob
        x_masked = x_batch.clone()
        x_masked[mask] = 0.0  # 将被掩盖的特征维度置为 0
        
        optimizer.zero_grad()
        encoded = encoder(x_masked)
        decoded = decoder(encoded)
        
        # 强制模型通过可见特征重构整个向量
        valid_mask = ~mask
        if valid_mask.sum() > 0:
            loss = criterion(decoded[valid_mask], x_batch[valid_mask])
            loss.backward()
            optimizer.step()

print("预训练完成，TRAIN 编码器已成功提取非线性潜变量。\n")

# 特征提取辅助函数：映射->编码
def extract_features(x_raw):
    encoder.eval()
    x_exp = expand_features(x_raw)
    with torch.no_grad():
        return encoder(torch.FloatTensor(x_exp)).numpy()

# ==============================================================================
# 2. 下游回归拟合阶段：拟合混合余弦函数
# ==============================================================================
print("===== 下游回归拟合与评估阶段 =====")

# 生成 300 个均匀分布的 x 值（替换随机采样，更适合捕捉周期性特征）
x_train = np.linspace(-0.5, 0.5, 300)  # 均匀分布更贴合余弦函数周期性
# 计算对应的真实 y 值：使用调整后的余弦函数（自然在[-0.5, 0.5]范围内）
y_train = 0.25 * np.cos(4 * np.pi * x_train) + 0.5 * x_train  # 核心：替换为余弦函数

# 划分训练集和测试集
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x_train.reshape(-1, 1),
    y_train,
    test_size=0.3,
    random_state=42
)

# 【核心结合点】：将原始 1D 数据经过扩展并提取为编码器的高维潜变量
x_train_enc = extract_features(x_train_raw)
x_test_enc = extract_features(x_test_raw)

# 指定模型路径
model_path = "tabpfn-v2-regressor.ckpt"

# 初始化 TabPFN 回归器
reg = TabPFNRegressor(model_path=model_path)
# 使用编码器提取的高维潜变量进行拟合
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
# 读取余弦函数真实数据文件（由之前的生成代码保存）
input_df = pd.read_excel('Cosx-True50.xlsx')
x_test_ext_raw = input_df.iloc[:, 0].values.reshape(-1, 1)  # 读取第一列 x 值

# 【核心结合点】：外部测试数据也必须经过同样的编码提取
x_test_ext_enc = extract_features(x_test_ext_raw)

# 进行预测
y_pred = reg.predict(x_test_ext_enc)

# 用预测值替换原 DataFrame 中的第二列 y 值
input_df.iloc[:, 1] = y_pred

# 保存预测结果
input_df.to_excel('Cosx-TRAIN50.xlsx', index=False)

# ==============================================================================
# 4. 读取与绘图 (完全保持原样)
# ==============================================================================
# 读取真实数据和预测数据
true_data = pd.read_excel('Cosx-True50.xlsx')
x_true = true_data.iloc[:, 0].values
y_true = true_data.iloc[:, 1].values

pred_data = pd.read_excel('Cosx-TRAIN50.xlsx')
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