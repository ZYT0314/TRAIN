import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
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

# 定义特征编码器（对齐ROC逻辑：轻量化设计）
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

# 定义特征解码器（用于自监督重建，对齐ROC逻辑）
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
        if self.matched_indices:
            if 'encoder.0.weight' in pretrain_dict and 'encoder.0.weight' in model_dict:
                pretrain_weight = pretrain_dict['encoder.0.weight'].clone()
                new_weight = model_dict['encoder.0.weight'].clone()
                for train_idx, pretrain_idx in enumerate(self.train_to_pretrain):
                    if pretrain_idx is not None and pretrain_idx < pretrain_weight.shape[1]:
                        new_weight[:, train_idx] = pretrain_weight[:, pretrain_idx]
                model_dict['encoder.0.weight'] = new_weight
                print(f"已迁移 {len(self.matched_indices)}/{len(self.train_to_pretrain)} 个匹配特征的第一层权重")
            
            for k in ['encoder.2.weight', 'encoder.2.bias']:
                if k in pretrain_dict and k in model_dict and pretrain_dict[k].shape == model_dict[k].shape:
                    model_dict[k] = pretrain_dict[k]
                    print(f"已迁移 {k} 权重")
        else:
            print("无匹配特征，不进行权重迁移，使用新初始化编码器")
        
        new_encoder.load_state_dict(model_dict)
        return new_encoder

    def transform(self, X_train, X_test):
        print(f"\n特征提取: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        X_train_matched = X_train[:, self.matched_indices] if self.matched_indices else np.array([])
        X_train_unmatched = X_train[:, self.unmatched_indices] if self.unmatched_indices else np.array([])
        X_test_matched = X_test[:, self.matched_indices] if self.matched_indices else np.array([])
        X_test_unmatched = X_test[:, self.unmatched_indices] if self.unmatched_indices else np.array([])
        
        if X_train_matched.size > 0:
            if self.pretrain_imputer.statistics_.shape[0] != X_train_matched.shape[1]:
                print(f"警告：预训练处理器特征维度({self.pretrain_imputer.statistics_.shape[0]})与匹配特征维度({X_train_matched.shape[1]})不匹配，使用训练集处理器处理匹配特征")
                X_train_matched = self.train_imputer.fit_transform(X_train_matched)
                X_train_matched = self.train_scaler.fit_transform(X_train_matched)
                X_test_matched = self.train_imputer.transform(X_test_matched)
                X_test_matched = self.train_scaler.transform(X_test_matched)
            else:
                X_train_matched = self.pretrain_imputer.transform(X_train_matched)
                X_train_matched = self.pretrain_scaler.transform(X_train_matched)
                X_test_matched = self.pretrain_imputer.transform(X_test_matched)
                X_test_matched = self.pretrain_scaler.transform(X_test_matched)
        
        if X_train_unmatched.size > 0:
            X_train_unmatched = self.train_imputer.fit_transform(X_train_unmatched)
            X_train_unmatched = self.train_scaler.fit_transform(X_train_unmatched)
            X_test_unmatched = self.train_imputer.transform(X_test_unmatched)
            X_test_unmatched = self.train_scaler.transform(X_test_unmatched)
        
        X_train_combined = np.hstack([
            X_train_matched if X_train_matched.size > 0 else np.zeros((X_train.shape[0], 0)),
            X_train_unmatched if X_train_unmatched.size > 0 else np.zeros((X_train.shape[0], 0))
        ])
        X_test_combined = np.hstack([
            X_test_matched if X_test_matched.size > 0 else np.zeros((X_test.shape[0], 0)),
            X_test_unmatched if X_test_unmatched.size > 0 else np.zeros((X_test.shape[0], 0))
        ])
        
        encoder = self._adapt_encoder(X_train_combined.shape[1])
        encoder.eval()
        with torch.no_grad():
            X_train_enc = encoder(torch.FloatTensor(X_train_combined).to(device)).cpu().numpy()
            X_test_enc = encoder(torch.FloatTensor(X_test_combined).to(device)).cpu().numpy()
        
        print(f"编码后维度: 训练集{X_train_enc.shape}, 测试集{X_test_enc.shape}")
        return X_train_enc, X_test_enc

# 掩码自监督预训练函数
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    pretrain_feature_names.remove('group')
    
    X_pretrain = pretrain_data.drop(columns=['group']).values
    imputer = SimpleImputer(strategy='mean')
    X_pretrain = imputer.fit_transform(X_pretrain)
    scaler = StandardScaler()
    X_pretrain = scaler.fit_transform(X_pretrain)
    
    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim).to(device)
    decoder = FeatureDecoder(input_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    mask_ratio_tensor = torch.tensor(mask_ratio, device=device)
    
    print(f"开始预训练 {input_dim} 维特征编码器...")
    for epoch in range(epochs):
        total_loss = 0
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_pretrain)), batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            x = batch[0].to(device)
            mask = torch.rand(x.shape, device=device) < mask_ratio_tensor
            x_masked = x.clone()
            x_masked[mask] = 0.0
            encoded = encoder(x_masked)
            decoded = decoder(encoded)
            loss = criterion(decoded, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    return encoder, scaler, imputer, pretrain_feature_names

# 主函数
def main():
    # 1. 预训练
    print("开始预训练...")
    encoder, scaler, imputer, pretrain_features = pretrain_feature_encoder(
        dataset_path='Pre-LCRA_VS_Control28.xlsx',
        epochs=50
    )
    
    # 2. 加载数据
    print("\n加载训练数据...")
    train_data = pd.read_excel('01RA_vs_Controlfeature28-bu.xlsx')
    train_features = [col for col in train_data.columns if col != 'group']
    y_internal = train_data['group'].map({'N': 0, 'T': 1}).values
    X_internal = train_data[train_features].values
    
    print("\n加载外部验证数据...")
    val_data = pd.read_excel('西南医院外部验证临床插补.xlsx')
    y_ext = val_data['group'].map({'N': 0, 'T': 1}).values
    X_ext = val_data[train_features].values

    # 3. 初始化特征提取器
    feature_extractor = UnifiedFeatureExtractor(
        encoder, scaler, imputer, pretrain_features, train_features
    )
    
    # 4. 定义模型
    models = [
        TabPFNClassifier(device=device, model_path="tabpfn-v3-classifier-v3_20260417_binary.ckpt"),
        CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, 
                           verbose=0, task_type="GPU" if device.type == 'cuda' else "CPU"),
        xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.05, n_estimators=100, 
                         max_depth=5, random_state=42, tree_method='gpu_hist' if device.type == 'cuda' else 'hist'),
        lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, n_estimators=100, 
                           max_depth=5, num_leaves=15, min_child_samples=5, min_split_gain=0.01, 
                           verbose=-1, random_state=42, device='gpu' if device.type == 'cuda' else 'cpu'),
        make_pipeline(StandardScaler(), SVC(probability=True, C=1.0, kernel='rbf', gamma='scale', random_state=42))
    ]
    
    model_names = ["TRAIN", "CatBoost", "XGBoost", "LightGBM", "SVM"]
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    
    # 5. 训练与外部验证
    all_mean_auprcs = []
    all_std_auprcs = []
    all_mean_precisions = []
    all_std_precisions = []
    
    for idx, model in enumerate(models):
        accuracies = []
        auprcs = []
        precisions_list = []
        recalls_list = []
        
        print(f"\n开始训练并对外部验证集进行测试 {model_names[idx]} 模型...")
        
        for i in range(10):
            print(f"迭代 {i+1}/10")
            # 内部划分仅用于提取训练子集
            X_train, _, y_train, _ = train_test_split(X_internal, y_internal, test_size=0.3, random_state=i)
            
            # 特征提取：处理【训练集】与【外部验证集】
            X_train_enc, X_ext_enc = feature_extractor.transform(X_train, X_ext)
            
            start_time = time.time()
            model.fit(X_train_enc, y_train)
            
            y_pred = model.predict(X_ext_enc)
            y_pred_proba = model.predict_proba(X_ext_enc)[:, 1]
            
            precision, recall, _ = precision_recall_curve(y_ext, y_pred_proba)
            auprc = auc(recall, precision)
            
            accuracies.append(accuracy_score(y_ext, y_pred))
            auprcs.append(auprc)
            precisions_list.append(precision)
            recalls_list.append(recall)
            print(f"迭代 {i+1} 外部验证 AUPRC: {auprc:.3f}")
        
        all_mean_auprcs.append(np.mean(auprcs))
        all_std_auprcs.append(np.std(auprcs))
        
        # 插值平均曲线
        mean_recall = np.linspace(0, 1, 100)
        interp_precisions = []
        for precision, recall in zip(precisions_list, recalls_list):
            idx_sort = np.argsort(recall)
            interp_precisions.append(np.interp(mean_recall, recall[idx_sort], precision[idx_sort]))
        
        all_mean_precisions.append(np.mean(interp_precisions, axis=0))
        all_std_precisions.append(np.std(interp_precisions, axis=0))
    
    # 打印对比结果
    desired_print_order = ["LightGBM", "TRAIN", "CatBoost", "XGBoost", "SVM"]
    print("\n===== 外部验证 AUPRC 性能对比 =====")
    for d_name in desired_print_order:
        idx = model_names.index(d_name)
        print(f"{d_name} Mean AUPRC: {all_mean_auprcs[idx]:.3f} (±{all_std_auprcs[idx]:.3f})")
        print("-" * 40)
    
    # 绘制外部验证 PRC 曲线 (严格按照 ROC 代码块的排版格式且不加对角虚线)
    plt.rcParams.update({
        'font.family': 'sans-serif',      # 指定使用无衬线字体家族
        'font.sans-serif': ['Arial'],     # 首选字体设为 Arial
        'pdf.fonttype': 42,                # 确保导出PDF时字体是可编辑的文本而非路径
        'ps.fonttype': 42,
        'font.size': 22,                  # 全局字体大小
        'axes.titlesize': 22,             # 标题字体大小
        'axes.labelsize': 22,             # 坐标轴标签字体大小
        'xtick.labelsize': 21,            # x轴刻度字体大小
        'ytick.labelsize': 21,            # y轴刻度字体大小
        'legend.fontsize': 18,    # 图例字体大小
    })
    
    plt.figure(figsize=(8, 6))
    for i in range(len(model_names)):
        plt.plot(mean_recall, all_mean_precisions[i], lw=2,
                 label=f'{model_names[i]} AUPRC: {all_mean_auprcs[i]:.3f} (±{all_std_auprcs[i]:.3f})',
                 color=colors[i])
        plt.fill_between(mean_recall, all_mean_precisions[i] - all_std_precisions[i],
                         all_mean_precisions[i] + all_std_precisions[i], alpha=0.08, color=colors[i])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Cross-centre Validation: Clincial (RA vs RA_DEP)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('external_validation_prc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()