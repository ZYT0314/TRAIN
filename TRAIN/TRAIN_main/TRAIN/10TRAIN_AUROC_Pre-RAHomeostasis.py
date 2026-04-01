import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.svm import SVC

# 设置CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 定义特征编码器（轻量化设计）
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2)  # 输出维度32
        )
        
    def forward(self, x):
        return self.encoder(x)

# 定义特征解码器（用于自监督重建）
class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)

# 特征提取器（核心修改：适配训练集维度，复用预训练权重）
class UnifiedFeatureExtractor:
    def __init__(self, pretrain_encoder, pretrain_feature_names, train_feature_names):
        self.pretrain_encoder = pretrain_encoder.to(device)
        self.pretrain_feature_names = pretrain_feature_names
        self.train_feature_names = train_feature_names
        self.hidden_dim = 64
        
        # 特征匹配映射（仅用于权重迁移，不强制维度匹配）
        self.train_to_pretrain = self._match_features()
        self.matched_indices = [i for i, idx in enumerate(self.train_to_pretrain) if idx is not None]
        
        # 训练集专用处理器（关键修改：不再复用预训练的imputer和scaler）
        self.train_imputer = SimpleImputer(strategy='mean')  # 训练集单独拟合
        self.train_scaler = StandardScaler()  # 训练集单独拟合
        
        # 匹配报告
        print("\n特征匹配报告:")
        matched_count = len(self.matched_indices)
        print(f"已匹配 {matched_count}/{len(train_feature_names)} 个特征（用于权重迁移）")
        unmatched_count = len(train_feature_names) - matched_count
        if unmatched_count > 0:
            print(f"未匹配 {unmatched_count} 个特征（使用随机初始化权重）")

    def _match_features(self):
        """匹配训练集特征与预训练特征，用于权重迁移"""
        pretrain_names = [name.strip().lower() for name in self.pretrain_feature_names]
        train_to_pretrain = []
        for train_feat in self.train_feature_names:
            train_feat_clean = train_feat.strip().lower()
            # 记录预训练特征索引（用于迁移权重）
            train_to_pretrain.append(
                pretrain_names.index(train_feat_clean) if train_feat_clean in pretrain_names else None
            )
        return train_to_pretrain

    def _adapt_encoder(self, train_feature_dim):
        """适配编码器输入维度，并迁移匹配特征的权重"""
        new_encoder = FeatureEncoder(input_dim=train_feature_dim, hidden_dim=self.hidden_dim).to(device)
        pretrain_dict = self.pretrain_encoder.state_dict()
        model_dict = new_encoder.state_dict()
        
        # 迁移第一层匹配特征的权重（关键修改：仅迁移存在匹配的特征权重）
        if 'encoder.0.weight' in pretrain_dict and 'encoder.0.weight' in model_dict:
            pretrain_weight = pretrain_dict['encoder.0.weight'].clone()  # 预训练第一层权重 (hidden_dim, 28)
            new_weight = model_dict['encoder.0.weight'].clone()  # 新编码器第一层权重 (hidden_dim, 10)
            
            # 仅迁移匹配的特征权重
            for train_idx, pretrain_idx in enumerate(self.train_to_pretrain):
                if pretrain_idx is not None and pretrain_idx < pretrain_weight.shape[1]:
                    new_weight[:, train_idx] = pretrain_weight[:, pretrain_idx]  # 迁移匹配特征的权重
            model_dict['encoder.0.weight'] = new_weight
            print(f"已迁移 {len(self.matched_indices)} 个匹配特征的第一层权重")
        
        # 迁移后续层权重（若形状匹配）
        for k in ['encoder.2.weight', 'encoder.2.bias']:
            if k in pretrain_dict and k in model_dict and pretrain_dict[k].shape == model_dict[k].shape:
                model_dict[k] = pretrain_dict[k]  # 迁移全连接层权重
                print(f"已迁移 {k} 权重")
        
        new_encoder.load_state_dict(model_dict)
        return new_encoder

    def transform(self, X_train, X_test):
        """处理训练集和测试集，使用训练集自己的imputer和scaler"""
        print(f"\n特征提取: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 处理缺失值（关键修改：用训练集拟合imputer）
        X_train_imputed = self.train_imputer.fit_transform(X_train)  # 训练集拟合+转换
        X_test_imputed = self.train_imputer.transform(X_test)  # 用训练集的imputer转换测试集
        
        # 标准化（关键修改：用训练集拟合scaler）
        X_train_scaled = self.train_scaler.fit_transform(X_train_imputed)  # 训练集拟合+转换
        X_test_scaled = self.train_scaler.transform(X_test_imputed)  # 用训练集的scaler转换测试集
        
        # 编码特征（适配10维输入）
        encoder = self._adapt_encoder(X_train_scaled.shape[1])  # 输入维度=10
        encoder.eval()
        with torch.no_grad():
            X_train_enc = encoder(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
            X_test_enc = encoder(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        
        print(f"编码后维度: 训练集{X_train_enc.shape}, 测试集{X_test_enc.shape}")
        return X_train_enc, X_test_enc

# 自监督预训练函数（保持不变，用28特征训练）
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    
    if 'group' in pretrain_feature_names:
        pretrain_feature_names.remove('group')
        X_pretrain = pretrain_data.drop(columns=['group']).values
    else:
        X_pretrain = pretrain_data.values
    
    # 预处理（仅用于预训练，不影响后续训练集）
    imputer_pretrain = SimpleImputer(strategy='mean')
    X_pretrain = imputer_pretrain.fit_transform(X_pretrain)
    scaler_pretrain = StandardScaler()
    X_pretrain = scaler_pretrain.fit_transform(X_pretrain)
    
    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim).to(device)
    decoder = FeatureDecoder(input_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    # 智能掩码
    feature_var = np.var(X_pretrain, axis=0)
    var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
    mask_prob = 0.1 + 0.3 * (1 - var_norm)
    mask_prob_tensor = torch.FloatTensor(mask_prob).to(device)
    print(f"预训练特征数: {input_dim}, 采用智能掩码策略")
    
    # 预训练循环
    print("开始预训练编码器...")
    for epoch in range(epochs):
        total_loss = 0
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_pretrain)), batch_size=batch_size, shuffle=True)
        
        for batch in dataloader:
            x = batch[0].to(device)
            mask = torch.rand(x.shape).to(device) < mask_prob_tensor[None, :]
            x_masked = x.clone()
            x_masked[mask] = 0.0
            
            encoded = encoder(x_masked)
            decoded = decoder(encoded)
            loss = criterion(decoded[~mask], x[~mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(X_pretrain)
            print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
    
    # 仅返回编码器和特征名（不再返回pretrain_imputer和scaler）
    return encoder, pretrain_feature_names

# 主函数
def main():
    # 1. 预训练（28特征，仅保留编码器和特征名）
    print("===== 预训练阶段 =====")
    encoder, pretrain_features = pretrain_feature_encoder(  # 关键修改：不再返回imputer和scaler
        dataset_path='Pre-LCRA_VS_RADEP39.xlsx',  # 28特征预训练数据
        epochs=50
    )
    
    # 2. 加载训练数据（10特征）
    print("\n===== 加载训练数据 =====")
    train_data = pd.read_excel('随访临床分析.xlsx')  # 10特征训练数据
    train_features = [col for col in train_data.columns if col != 'group']
    y = train_data['group'].map({'N': 0, 'T': 1}).values
    X = train_data[train_features].values
    print(f"训练集特征数: {len(train_features)}（目标维度）")
    
    # 3. 初始化特征提取器（关键修改：不再传入pretrain_imputer和scaler）
    feature_extractor = UnifiedFeatureExtractor(
        encoder, pretrain_features, train_features
    )
    
    # 4. 定义模型
    models = [
        TabPFNClassifier(device=device, model_path="tabpfn-v2-classifier.ckpt"),
        CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, 
                           verbose=0, task_type="GPU" if device.type == 'cuda' else "CPU"),
        xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.05, n_estimators=100, 
                         max_depth=5, random_state=42, tree_method='gpu_hist' if device.type == 'cuda' else 'hist'),
        lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=50, learning_rate=0.05, 
                          n_estimators=100, min_split_gain=0.01, random_state=42,
                          device='gpu' if device.type == 'cuda' else 'cpu'),
        SVC(probability=True, C=0.001, kernel='linear', gamma='scale', random_state=42)
    ]
    
    original_model_names = ["TRAIN", "CatBoost", "XGBoost", "LightGBM", "SVM"]
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    
    # 5. 训练与评估
    all_mean_aucs = []
    all_std_aucs = []
    all_mean_tprs = []
    all_std_tprs = []
    all_mean_accuracies = []
    all_std_accuracies = []
    
    for idx, model in enumerate(models):
        accuracies = []
        aucs = []
        fprs = []
        tprs = []
        
        print(f"\n===== 训练 {original_model_names[idx]} 模型 =====")
        
        for i in range(10):
            print(f"\n迭代 {i+1}/10")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
            print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            # 特征处理（使用训练集自己的imputer和scaler）
            X_train_enc, X_test_enc = feature_extractor.transform(X_train, X_test)
            
            start_time = time.time()
            model.fit(X_train_enc, y_train)
            print(f"训练时间: {time.time() - start_time:.2f}秒")
            
            y_pred = model.predict(X_test_enc)
            y_pred_proba = model.predict_proba(X_test_enc)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracies.append(accuracy)
            aucs.append(auc)
            print(f"迭代 {i+1} 性能: 准确率={accuracy:.3f}, AUC={auc:.3f}")
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fprs.append(fpr)
            tprs.append(tpr)
        
        # 计算平均指标
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        all_mean_accuracies.append(mean_accuracy)
        all_std_accuracies.append(std_accuracy)
        all_mean_aucs.append(mean_auc)
        all_std_aucs.append(std_auc)
        
        # 插值ROC曲线
        mean_fpr = np.linspace(0, 1, 100)
        tprs_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)]
        mean_tpr = np.mean(tprs_interp, axis=0)
        std_tpr = np.std(tprs_interp, axis=0)
        
        all_mean_tprs.append(mean_tpr)
        all_std_tprs.append(std_tpr)
    
    # 打印最终性能对比
    print("\n===== 模型性能对比 =====")
    for idx, name in enumerate(original_model_names):
        print(f"{name}:")
        print(f"  平均准确率: {all_mean_accuracies[idx]:.3f} (±{all_std_accuracies[idx]:.3f})")
        print(f"  平均AUC: {all_mean_aucs[idx]:.3f} (±{all_std_aucs[idx]:.3f})")
        print("-" * 40)
    
        # 绘制ROC曲线
    plt.rcParams.update({
        'font.size': 22,          # 全局字体大小
        'axes.titlesize': 22,     # 标题字体大小
        'axes.labelsize': 22,     # 坐标轴标签字体大小
        'xtick.labelsize': 21,    # x轴刻度字体大小
        'ytick.labelsize': 21,    # y轴刻度字体大小
        'legend.fontsize': 18,    # 图例字体大小
    })
    
    plt.figure(figsize=(8, 6))
    for i in range(len(original_model_names)):
        plt.plot(mean_fpr, all_mean_tprs[i], lw=2,
                 label=f'{original_model_names[i]} AUROC: {all_mean_aucs[i]:.3f} (±{all_std_aucs[i]:.3f})',
                 color=colors[i])
        plt.fill_between(mean_fpr, all_mean_tprs[i] - all_std_tprs[i],
                         all_mean_tprs[i] + all_std_tprs[i], alpha=0.08, color=colors[i])
    
    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Clinical Models')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('comparison_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()