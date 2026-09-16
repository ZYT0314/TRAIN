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
# 1. 增强版 TRAIN 模型：通用多维扩展 + 自监督掩码预训练 (PoC 验证)
# ==============================================================================
print("===== TRAIN 模型自监督预训练阶段 =====")

# 将 1D 扩展为 5D 通用数学基底，模拟高维空间的互相影响（保持框架绝对通用性）
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

print("预训练完成，TRAIN 编码器已成功提取通用潜变量。\n")

# 特征提取辅助函数：映射->编码
def extract_features(x_raw):
    encoder.eval()
    x_exp = expand_features(x_raw)
    with torch.no_grad():
        return encoder(torch.FloatTensor(x_exp)).numpy()


# ==============================================================================
# 2. 下游回归拟合阶段：拟合阶梯函数
# ==============================================================================
print("===== 下游回归拟合与评估阶段 =====")

# 定义区间边界
x_boundaries = np.linspace(-0.5, 0.5, 6)
# 每个阶梯对应60个数据点
points_per_step = 60
total_points = len(x_boundaries) * points_per_step

# 生成均匀分布的x值 (阶梯函数)
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
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x.reshape(-1, 1),
    y,
    test_size=0.3,
    random_state=42
)

# 【核心结合点】：将原始 1D 数据经过扩展并提取为编码器的高维潜变量
x_train_enc = extract_features(x_train_raw)
x_test_enc = extract_features(x_test_raw)

# 指定模型路径
model_path = "tabpfn-v2-regressor.ckpt"

# 初始化TabPFN回归器
reg = TabPFNRegressor(model_path=model_path)
# 使用编码器提取的高维潜变量进行拟合
reg.fit(x_train_enc, y_train)

# Predict a point estimate (using the mean)
predictions = reg.predict(x_test_enc)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# Predict quantiles
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
# 读取包含x和y值的数据表格
try:
    input_df = pd.read_excel('step_function25.xlsx')
    x_test_ext_raw = input_df.iloc[:, 0].values.reshape(-1, 1)  # 读取第一列x值
    
    # 【核心结合点】：外部测试数据也必须经过同样的编码提取
    x_test_ext_enc = extract_features(x_test_ext_raw)
    
    # 进行预测
    y_pred_ext = reg.predict(x_test_ext_enc)
    
    # 用预测值替换原DataFrame中的第二列y值
    input_df.iloc[:, 1] = y_pred_ext
    
    # 保存结果到新的Excel文件
    input_df.to_excel('step_function25-TRAIN.xlsx', index=False)
    
    # ==============================================================================
    # 4. 读取与绘图 (完全保持原样)
    # ==============================================================================
    # 读取真实数据
    true_data = pd.read_excel('step_function25.xlsx')
    x_true = true_data.iloc[:, 0].values
    y_true = true_data.iloc[:, 1].values
    
    # 读取预测数据
    pred_data = pd.read_excel('step_function25-TRAIN.xlsx')
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
    x_boundaries_plot = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    
    # 绘制真实数据曲线图（橙色），线宽3
    for x_seg, y_seg in zip(true_segments_x, true_segments_y):
        plt.plot(x_seg, y_seg, color='orange', label='True', linewidth=3)
    
    # 绘制预测数据曲线（蓝色），线宽1.5
    for i in range(len(x_boundaries_plot) - 1):
        start_bound = x_boundaries_plot[i]
        end_bound = x_boundaries_plot[i + 1]
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
    
    # 图例字体大小设置为18
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=18)
    
    plt.tight_layout()  # 优化布局
    plt.show()

except FileNotFoundError:
    print("提示：未找到本地的 'step_function25.xlsx' 文件，跳过绘图步骤。模型已完成评估！")