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
from sklearn.pipeline import make_pipeline

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

# 特征提取器（处理特征匹配和权重迁移）
class UnifiedFeatureExtractor:
    def __init__(self, pretrain_encoder, pretrain_scaler, pretrain_imputer, 
                 pretrain_feature_names, train_feature_names):
        self.pretrain_encoder = pretrain_encoder.to(device)
        self.pretrain_scaler = pretrain_scaler
        self.pretrain_imputer = pretrain_imputer
        self.pretrain_feature_names = pretrain_feature_names
        self.train_feature_names = train_feature_names
        self.hidden_dim = 64
        
        # 特征匹配映射
        self.train_to_pretrain = self._match_features()
        self.matched_indices = [i for i, idx in enumerate(self.train_to_pretrain) if idx is not None]
        self.unmatched_indices = [i for i, idx in enumerate(self.train_to_pretrain) if idx is None]
        
        # 训练集专用处理器
        self.train_imputer = SimpleImputer(strategy='mean')
        self.train_scaler = StandardScaler()
        
        # 匹配报告
        print("\n特征匹配报告:")
        matched_count = len(self.matched_indices)
        print(f"已匹配 {matched_count}/{len(train_feature_names)} 个特征")
        if self.unmatched_indices:
            print(f"未匹配 {len(self.unmatched_indices)} 个特征 (示例):")
            for i in self.unmatched_indices[:5]:
                print(f"  - '{self.train_feature_names[i]}'")

    def _match_features(self):
        pretrain_names = [name.strip().lower() for name in self.pretrain_feature_names]
        train_to_pretrain = []
        for train_feat in self.train_feature_names:
            train_feat_clean = train_feat.strip().lower()
            train_to_pretrain.append(pretrain_names.index(train_feat_clean) if train_feat_clean in pretrain_names else None)
        return train_to_pretrain

    def _adapt_encoder(self, train_feature_dim):
        new_encoder = FeatureEncoder(input_dim=train_feature_dim, hidden_dim=self.hidden_dim).to(device)
        pretrain_dict = self.pretrain_encoder.state_dict()
        model_dict = new_encoder.state_dict()
        
        # 仅当有匹配特征时才迁移权重
        if self.matched_indices:  # 新增：判断是否有匹配特征
            # 迁移第一层匹配特征权重
            if 'encoder.0.weight' in pretrain_dict and 'encoder.0.weight' in model_dict:
                pretrain_weight = pretrain_dict['encoder.0.weight'].clone()
                new_weight = model_dict['encoder.0.weight'].clone()
                for train_idx, pretrain_idx in enumerate(self.train_to_pretrain):
                    if pretrain_idx is not None and pretrain_idx < pretrain_weight.shape[1]:
                        new_weight[:, train_idx] = pretrain_weight[:, pretrain_idx]
                model_dict['encoder.0.weight'] = new_weight
                print(f"已迁移 {len(self.matched_indices)}/{len(self.train_to_pretrain)} 个匹配特征的第一层权重")
            
            # 迁移后续层权重
            for k in ['encoder.2.weight', 'encoder.2.bias']:
                if k in pretrain_dict and k in model_dict and pretrain_dict[k].shape == model_dict[k].shape:
                    model_dict[k] = pretrain_dict[k]
                    print(f"已迁移 {k} 权重")
        else:
            print("无匹配特征，不进行权重迁移，使用新初始化编码器")  # 新增：无匹配特征时提示
        
        new_encoder.load_state_dict(model_dict)
        return new_encoder

    def transform(self, X_train, X_test):
        print(f"\n特征提取: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 分离匹配/未匹配特征
        X_train_matched = X_train[:, self.matched_indices] if self.matched_indices else np.array([])
        X_train_unmatched = X_train[:, self.unmatched_indices] if self.unmatched_indices else np.array([])
        X_test_matched = X_test[:, self.matched_indices] if self.matched_indices else np.array([])
        X_test_unmatched = X_test[:, self.unmatched_indices] if self.unmatched_indices else np.array([])
        
        # 处理匹配特征：仅当有匹配特征时使用预训练处理器
        if X_train_matched.size > 0:
            # 新增：检查预训练处理器的特征维度是否与匹配特征一致
            if self.pretrain_imputer.statistics_.shape[0] != X_train_matched.shape[1]:
                print(f"警告：预训练处理器特征维度({self.pretrain_imputer.statistics_.shape[0]})与匹配特征维度({X_train_matched.shape[1]})不匹配，使用训练集处理器处理匹配特征")
                # 改用训练集处理器处理匹配特征（避免维度不匹配错误）
                X_train_matched = self.train_imputer.fit_transform(X_train_matched)
                X_train_matched = self.train_scaler.fit_transform(X_train_matched)
                X_test_matched = self.train_imputer.transform(X_test_matched)
                X_test_matched = self.train_scaler.transform(X_test_matched)
            else:
                # 维度一致时正常使用预训练处理器
                X_train_matched = self.pretrain_imputer.transform(X_train_matched)
                X_train_matched = self.pretrain_scaler.transform(X_train_matched)
                X_test_matched = self.pretrain_imputer.transform(X_test_matched)
                X_test_matched = self.pretrain_scaler.transform(X_test_matched)
        
        # 处理未匹配特征
        if X_train_unmatched.size > 0:
            X_train_unmatched = self.train_imputer.fit_transform(X_train_unmatched)
            X_train_unmatched = self.train_scaler.fit_transform(X_train_unmatched)
            X_test_unmatched = self.train_imputer.transform(X_test_unmatched)
            X_test_unmatched = self.train_scaler.transform(X_test_unmatched)
        
        # 合并特征
        X_train_combined = np.hstack([
            X_train_matched if X_train_matched.size > 0 else np.zeros((X_train.shape[0], 0)),
            X_train_unmatched if X_train_unmatched.size > 0 else np.zeros((X_train.shape[0], 0))
        ])
        X_test_combined = np.hstack([
            X_test_matched if X_test_matched.size > 0 else np.zeros((X_test.shape[0], 0)),
            X_test_unmatched if X_test_unmatched.size > 0 else np.zeros((X_test.shape[0], 0))
        ])
        
        # 编码特征
        encoder = self._adapt_encoder(X_train_combined.shape[1])
        encoder.eval()
        with torch.no_grad():
            X_train_enc = encoder(torch.FloatTensor(X_train_combined).to(device)).cpu().numpy()
            X_test_enc = encoder(torch.FloatTensor(X_test_combined).to(device)).cpu().numpy()
        
        print(f"编码后维度: 训练集{X_train_enc.shape}, 测试集{X_test_enc.shape}")
        return X_train_enc, X_test_enc

# 自监督预训练函数
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    
    if 'group' in pretrain_feature_names:
        pretrain_feature_names.remove('group')
        X_pretrain = pretrain_data.drop(columns=['group']).values
    else:
        X_pretrain = pretrain_data.values
    
    # 预处理
    imputer = SimpleImputer(strategy='mean')
    X_pretrain = imputer.fit_transform(X_pretrain)
    scaler = StandardScaler()
    X_pretrain = scaler.fit_transform(X_pretrain)
    
    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim).to(device)
    decoder = FeatureDecoder(input_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
     # 智能掩码（修复类型不匹配问题）
    feature_var = np.var(X_pretrain, axis=0)
    var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
    mask_prob = 0.1 + 0.3 * (1 - var_norm)
    mask_prob_tensor = torch.FloatTensor(mask_prob).to(device)  # 转换为张量并移至设备
    print(f"预训练特征数: {input_dim}, 采用智能掩码策略")
    
    # 预训练循环
    print("开始预训练编码器...")
    for epoch in range(epochs):
        total_loss = 0
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_pretrain)), batch_size=batch_size, shuffle=True)
        
        for batch in dataloader:
            x = batch[0].to(device)
            # 修复：使用PyTorch张量进行比较
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
    
    return encoder, scaler, imputer, pretrain_feature_names

# 主函数
def main():
    # 1. 预训练
    print("===== 预训练阶段 =====")
    encoder, scaler, imputer, pretrain_features = pretrain_feature_encoder(
        dataset_path='Pre-LCRA_VS_Control28.xlsx',  # 更新为你的预训练数据文件
        epochs=50
    )
    
    # 加载训练数据
    print("\n加载训练数据...")
    train_data = pd.read_excel('01RA_vs_Controlfeature28-bu.xlsx')  # 替换为你的训练数据路径
    train_features = [col for col in train_data.columns if col != 'group']
    y = train_data['group'].map({'N': 0, 'T': 1}).values
    X = train_data[train_features].values
    
    # 3. 初始化特征提取器
    feature_extractor = UnifiedFeatureExtractor(
        encoder, scaler, imputer, pretrain_features, train_features
    )
    
    # 4. 定义模型
    models = [
        TabPFNClassifier(device=device, model_path="tabpfn-v2-classifier.ckpt"),
        CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, 
                           verbose=0, task_type="GPU" if device.type == 'cuda' else "CPU"),
        xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.05, n_estimators=100, 
                         max_depth=5, random_state=42, tree_method='gpu_hist' if device.type == 'cuda' else 'hist'),
        lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, n_estimators=100, 
                           max_depth=5, num_leaves=15, min_child_samples=5, min_split_gain=0.01, 
                           verbose=-1, random_state=42, device='gpu' if device.type == 'cuda' else 'cpu'),
        make_pipeline(StandardScaler(), SVC(probability=True, C=1.0, kernel='rbf', gamma='scale', random_state=42))
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
    
    # 绘制ROC曲线（应用指定的字体设置）
    plt.rcParams.update({
        'font.family': 'sans-serif',      # 指定使用无衬线字体家族
        'font.sans-serif': ['Arial'],     # 首选字体设为 Arial
        'pdf.fonttype': 42,               # 确保导出PDF时字体是可编辑的文本而非路径
        'ps.fonttype': 42,
        'font.size': 22,                  # 全局字体大小
        'axes.titlesize': 22,             # 标题字体大小
        'axes.labelsize': 22,             # 坐标轴标签字体大小
        'xtick.labelsize': 21,            # x轴刻度字体大小
        'ytick.labelsize': 21,            # y轴刻度字体大小
        'legend.fontsize': 18,    # 图例字体大小
    })
    
    plt.figure(figsize=(8, 6))
    for i in range(len(original_model_names)):
        plt.plot(mean_fpr, all_mean_tprs[i], lw=2,
                 label=f'{original_model_names[i]} AUROC: {all_mean_aucs[i]:.3f} (±{all_std_aucs[i]:.3f})',
                 color=colors[i])
        plt.fill_between(mean_fpr, all_mean_tprs[i] - all_std_tprs[i],
                         all_mean_tprs[i] + all_std_tprs[i], alpha=0.08, color=colors[i])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Diagnosis: Clinical (RA vs HC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('comparison_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()