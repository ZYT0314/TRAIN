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

# 多组学融合特征提取器（核心：仅对匹配特征迁移学习，不匹配特征用原始处理）
class MultiOmicsFeatureExtractor:
    def __init__(self, pretrain_results, train_feature_names):
        self.omics_extractors = []
        self.train_feature_names = train_feature_names
        
        # 存储每个组学的预训练信息
        for encoder, scaler, imputer, pretrain_feats, omics_name in pretrain_results:
            extractor = {
                'encoder': encoder.to(device),
                'scaler': scaler,
                'imputer': imputer,
                'pretrain_feats': [f.strip().lower() for f in pretrain_feats],
                'name': omics_name,
                'pretrain_dim': len(pretrain_feats)  # 记录预训练时的特征维度
            }
            self.omics_extractors.append(extractor)
        
        # 为每个组学的匹配特征创建单独的特征匹配器
        self.group_imputers = {extractor['name']: SimpleImputer(strategy='mean') 
                              for extractor in self.omics_extractors}
        self.group_scalers = {extractor['name']: StandardScaler() 
                             for extractor in self.omics_extractors}
        self.common_imputer = SimpleImputer(strategy='mean')
        self.common_scaler = StandardScaler()
        
        # 特征匹配报告
        self._report_feature_matching()

    def _report_feature_matching(self):
        """报告每个组学与融合数据的特征匹配情况"""
        print("\n多组学特征匹配报告:")
        for extractor in self.omics_extractors:
            matched = [f for f in self.train_feature_names 
                      if f.strip().lower() in extractor['pretrain_feats']]
            print(f"{extractor['name']} 匹配特征: {len(matched)}/{len(self.train_feature_names)}")

    def _get_matched_features(self, X, extractor, is_train=True):
        # 提取与当前组学匹配的特征
        matched_indices = [i for i, f in enumerate(self.train_feature_names)
                          if f.strip().lower() in extractor['pretrain_feats']]
        if not matched_indices:
            return np.array([]), matched_indices
        
        X_matched = X[:, matched_indices]
        
        # 检查特征维度是否匹配
        if X_matched.shape[1] != extractor['pretrain_dim']:
            print(f"{extractor['name']} 特征维度不匹配 ({X_matched.shape[1]} vs {extractor['pretrain_dim']})，使用新处理器")
            if is_train:
                X_matched = self.group_imputers[extractor['name']].fit_transform(X_matched)
                X_matched = self.group_scalers[extractor['name']].fit_transform(X_matched)
            else:
                X_matched = self.group_imputers[extractor['name']].transform(X_matched)
                X_matched = self.group_scalers[extractor['name']].transform(X_matched)
        else:
            X_matched = extractor['imputer'].transform(X_matched)
            X_matched = extractor['scaler'].transform(X_matched)
            
        return X_matched, matched_indices

    def _get_unmatched_features(self, X, matched_indices_all, is_train=True):
        # 提取所有组学都未匹配的特征
        all_matched = set()
        for indices in matched_indices_all:
            all_matched.update(indices)
        unmatched_indices = [i for i in range(X.shape[1]) if i not in all_matched]
        
        if not unmatched_indices:
            return np.array([])
            
        X_unmatched = X[:, unmatched_indices]
        if is_train:
            X_unmatched = self.common_imputer.fit_transform(X_unmatched)
            X_unmatched = self.common_scaler.fit_transform(X_unmatched)
        else:
            X_unmatched = self.common_imputer.transform(X_unmatched)
            X_unmatched = self.common_scaler.transform(X_unmatched)
        return X_unmatched

    def transform(self, X_train, X_test):
        print(f"\n多组学特征提取: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        train_encoded_features = []
        test_encoded_features = []
        matched_indices_all = []
        
        for extractor in self.omics_extractors:
            X_train_matched, matched_indices = self._get_matched_features(
                X_train, extractor, is_train=True)
            X_test_matched, _ = self._get_matched_features(
                X_test, extractor, is_train=False)
                
            matched_indices_all.append(matched_indices)
            
            if X_train_matched.size == 0:
                print(f"{extractor['name']} 无匹配特征，跳过编码")
                continue
                
            current_encoder = extractor['encoder']
            if hasattr(current_encoder.encoder[0], 'in_features') and \
               current_encoder.encoder[0].in_features != X_train_matched.shape[1]:
                print(f"{extractor['name']} 编码器输入维度不匹配，创建适配编码器")
                adapted_encoder = FeatureEncoder(X_train_matched.shape[1]).to(device)
                pretrained_dict = current_encoder.state_dict()
                model_dict = adapted_encoder.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                                  k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                adapted_encoder.load_state_dict(model_dict)
                current_encoder = adapted_encoder
            
            current_encoder.eval()
            with torch.no_grad():
                X_train_enc = current_encoder(torch.FloatTensor(X_train_matched).to(device)).cpu().numpy()
                X_test_enc = current_encoder(torch.FloatTensor(X_test_matched).to(device)).cpu().numpy()
            
            train_encoded_features.append(X_train_enc)
            test_encoded_features.append(X_test_enc)
            print(f"{extractor['name']} 编码后维度: 训练集{X_train_enc.shape}, 测试集{X_test_enc.shape}")
        
        X_train_unmatched = self._get_unmatched_features(X_train, matched_indices_all, is_train=True)
        X_test_unmatched = self._get_unmatched_features(X_test, matched_indices_all, is_train=False)
        
        if X_train_unmatched.size > 0:
            train_encoded_features.append(X_train_unmatched)
            test_encoded_features.append(X_test_unmatched)
            print(f"未匹配特征维度: 训练集{X_train_unmatched.shape}, 测试集{X_test_unmatched.shape}")
        
        X_train_combined = np.hstack(train_encoded_features) if train_encoded_features else X_train
        X_test_combined = np.hstack(test_encoded_features) if test_encoded_features else X_test
        
        print(f"最终融合特征维度: 训练集{X_train_combined.shape}, 测试集{X_test_combined.shape}")
        return X_train_combined, X_test_combined

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
    
    # 智能掩码策略
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
    
    return encoder, scaler, imputer, pretrain_feature_names

# 主函数
def main():
    # 1. 三个组学分别预训练
    print("===== 多组学联合预训练阶段 =====")
    pretrain_file_paths = [
        'Pre-LCRA_VS_RADEP39.xlsx',   # 组学1
        'Pre-RABC_RADEP153.xlsx',    # 组学2
        'Pre-RADB204样本.xlsx'          # 组学3
    ]
    pretrain_results = []
    
    for i, file_path in enumerate(pretrain_file_paths, 1):
        print(f"\n----- 开始第 {i} 个组学预训练（文件：{file_path}）-----")
        encoder, scaler, imputer, pretrain_features = pretrain_feature_encoder(
            dataset_path=file_path,
            epochs=50
        )
        pretrain_results.append( (encoder, scaler, imputer, pretrain_features, f"组学{i}") )
        print(f"----- 第 {i} 个组学预训练完成 -----")
    
    # 2. 加载融合分析数据
    print("\n===== 加载融合分析数据 =====")
    train_data = pd.read_excel('11RA_DEP临床转录蛋白合并新.xlsx')
    train_features = [col for col in train_data.columns if col != 'group']
    
    if 'group' not in train_data.columns:
        raise ValueError("融合数据必须包含'group'列作为标签")
    y = train_data['group'].map({'N': 0, 'T': 1}).values
    X = train_data[train_features].values
    
    # 3. 初始化多组学融合特征提取器
    print("\n===== 初始化多组学融合特征提取器 =====")
    feature_extractor = MultiOmicsFeatureExtractor(
        pretrain_results=pretrain_results,
        train_feature_names=train_features
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
    all_mean_auprcs = []
    all_std_auprcs = []
    all_mean_precisions = []
    all_std_precisions = []
    mean_recall = np.linspace(0, 1, 100)
    
    for idx, model in enumerate(models):
        accuracies = []
        auprcs = []
        precisions_list = []
        recalls_list = []
        
        print(f"\n===== 训练 {original_model_names[idx]} 模型 =====")
        
        for i in range(10):
            print(f"\n迭代 {i+1}/10")
            # 严格保持：全员统一使用普通随机抽样
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=i)
            print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            # 严格保持：全员统一使用特征提取
            X_train_enc, X_test_enc = feature_extractor.transform(X_train, X_test)
            
            start_time = time.time()
            # 严格保持：全员直接拟合
            model.fit(X_train_enc, y_train)
            print(f"训练时间: {time.time() - start_time:.2f}秒")
            
            y_pred = model.predict(X_test_enc)
            y_pred_proba = model.predict_proba(X_test_enc)[:, 1]
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = auc(recall, precision)
            auprcs.append(auprc)
            print(f"迭代 {i+1} 性能: 准确率={accuracy:.3f}, AUPRC={auprc:.3f}")
            
            precisions_list.append(precision)
            recalls_list.append(recall)
        
        # 计算平均指标
        all_mean_auprcs.append(np.mean(auprcs))
        all_std_auprcs.append(np.std(auprcs))
        
        # PRC曲线插值
        interp_precisions = []
        for precision, recall in zip(precisions_list, recalls_list):
            idx_sort = np.argsort(recall)
            interp_precisions.append(np.interp(mean_recall, recall[idx_sort], precision[idx_sort]))
        
        all_mean_precisions.append(np.mean(interp_precisions, axis=0))
        all_std_precisions.append(np.std(interp_precisions, axis=0))
    
    # 打印最终对比
    print("\n===== 多组学融合模型性能对比 =====")
    for idx, name in enumerate(original_model_names):
        print(f"{name}:")
        print(f"  平均AUPRC: {all_mean_auprcs[idx]:.3f} (±{all_std_auprcs[idx]:.3f})")
        print("-" * 40)
    
    # 绘制PRC曲线
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 18,
    })
    
    plt.figure(figsize=(8, 6))
    for i in range(len(original_model_names)):
        plt.plot(mean_recall, all_mean_precisions[i], lw=2,
                 label=f'{original_model_names[i]} AUPRC: {all_mean_auprcs[i]:.3f} (±{all_std_auprcs[i]:.3f})',
                 color=colors[i])
        plt.fill_between(mean_recall, all_mean_precisions[i] - all_std_precisions[i],
                         all_mean_precisions[i] + all_std_precisions[i], alpha=0.08, color=colors[i])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Stratification: Multi-Omics (RA vs RA_DEP)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('multi_omics_fusion_prc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()