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
from sklearn.calibration import CalibratedClassifierCV  # 从第一个代码添加SVM概率校准校准所需模块

# 设置CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 定义特征编码器（保持第二个代码的轻量化设计）
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

# 定义特征解码器（用于自监督重建，保持第二个代码不变）
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

# 多组学融合特征提取器（保持第二个代码的预训练逻辑）
class MultiOmicsFeatureExtractor:
    def __init__(self, pretrain_results, train_feature_names):
        self.omics_extractors = []
        self.train_feature_names = train_feature_names
        
        # 为每个组学创建单独的特征匹配器
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
        
        # 训练集专用处理器（保持第二个代码不变）
        self.group_imputers = {extractor['name']: SimpleImputer(strategy='mean') 
                              for extractor in self.omics_extractors}
        self.group_scalers = {extractor['name']: StandardScaler() 
                             for extractor in self.omics_extractors}
        self.common_imputer = SimpleImputer(strategy='mean')
        self.common_scaler = StandardScaler()
        
        # 特征匹配报告
        self._report_feature_matching()

    def _report_feature_matching(self):
        print("\n多组学特征匹配报告:")
        for extractor in self.omics_extractors:
            matched = [f for f in self.train_feature_names 
                      if f.strip().lower() in extractor['pretrain_feats']]
            print(f"{extractor['name']} 匹配特征: {len(matched)}/{len(self.train_feature_names)}")

    def _get_matched_features(self, X, extractor, is_train=True):
        # 提取与当前组学匹配的特征（保持第二个代码逻辑）
        matched_indices = [i for i, f in enumerate(self.train_feature_names)
                          if f.strip().lower() in extractor['pretrain_feats']]
        if not matched_indices:
            return np.array([]), matched_indices
        
        X_matched = X[:, matched_indices]
        
        # 检查特征维度是否匹配（保持第二个代码逻辑）
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
        # 提取所有组学都未匹配的特征（保持第二个代码逻辑）
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
        
        # 存储每个组学的编码特征（保持第二个代码逻辑）
        train_encoded_features = []
        test_encoded_features = []
        matched_indices_all = []
        
        # 每个组学分别处理匹配特征
        for extractor in self.omics_extractors:
            X_train_matched, matched_indices = self._get_matched_features(
                X_train, extractor, is_train=True)
            X_test_matched, _ = self._get_matched_features(
                X_test, extractor, is_train=False)
                
            matched_indices_all.append(matched_indices)
            
            if X_train_matched.size == 0:
                print(f"{extractor['name']} 无匹配特征，跳过编码")
                continue
                
            # 适配编码器输入维度（保持第二个代码逻辑）
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
        
        # 处理未匹配特征（保持第二个代码逻辑）
        X_train_unmatched = self._get_unmatched_features(X_train, matched_indices_all, is_train=True)
        X_test_unmatched = self._get_unmatched_features(X_test, matched_indices_all, is_train=False)
        
        # 合并所有特征
        all_train_features = train_encoded_features
        all_test_features = test_encoded_features
        
        if X_train_unmatched.size > 0:
            all_train_features.append(X_train_unmatched)
            all_test_features.append(X_test_unmatched)
            print(f"未匹配特征维度: 训练集{X_train_unmatched.shape}, 测试集{X_test_unmatched.shape}")
        
        # 最终特征拼接（为SVM添加第一个代码的二次标准化，不影响其他模型）
        X_train_combined = np.hstack(all_train_features) if all_train_features else X_train
        X_test_combined = np.hstack(all_test_features) if all_test_features else X_test
        
        # 仅为SVM添加的特征标准化（从第一个代码迁移）
        self.svm_scaler = None
        if X_train_combined.size > 0:
            self.svm_scaler = StandardScaler()
            X_train_combined_svm = self.svm_scaler.fit_transform(X_train_combined)
            X_test_combined_svm = self.svm_scaler.transform(X_test_combined)
        else:
            X_train_combined_svm = X_train_combined
            X_test_combined_svm = X_test_combined
        
        # 返回两种特征：SVM专用（标准化）和其他模型用（原始）
        return X_train_combined, X_test_combined, X_train_combined_svm, X_test_combined_svm

# 自监督预训练函数（保持第二个代码的智能掩码逻辑不变）
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    
    if 'group' in pretrain_feature_names:
        pretrain_feature_names.remove('group')
        X_pretrain = pretrain_data.drop(columns=['group']).values
    else:
        X_pretrain = pretrain_data.values
    
    # 预处理（保持第二个代码逻辑）
    imputer = SimpleImputer(strategy='mean')
    X_pretrain = imputer.fit_transform(X_pretrain)
    scaler = StandardScaler()
    X_pretrain = scaler.fit_transform(X_pretrain)
    
    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim).to(device)
    decoder = FeatureDecoder(input_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    # 智能掩码（保持第二个代码的独特逻辑）
    feature_var = np.var(X_pretrain, axis=0)
    var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
    mask_prob = 0.1 + 0.3 * (1 - var_norm)
    mask_prob_tensor = torch.FloatTensor(mask_prob).to(device)
    print(f"预训练特征数: {input_dim}, 采用智能掩码策略")
    
    # 预训练循环（保持第二个代码逻辑）
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

# 主函数（仅替换SVM训练逻辑，其他模型保持第二个代码逻辑）
def main():
    # 1. 三个组学分别预训练（保持第二个代码逻辑）
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
    
    # 2. 加载融合分析数据（保持第二个代码逻辑）
    print("\n===== 加载融合分析数据 =====")
    train_data = pd.read_excel('RA_DEP临床转录蛋白合并新.xlsx')
    train_features = [col for col in train_data.columns if col != 'group']
    
    if 'group' not in train_data.columns:
        raise ValueError("融合数据必须包含'group'列作为标签")
    y = train_data['group'].map({'N': 0, 'T': 1}).values
    X = train_data[train_features].values
    
    # 3. 初始化多组学融合特征提取器（保持第二个代码逻辑）
    print("\n===== 初始化多组学融合特征提取器 =====")
    feature_extractor = MultiOmicsFeatureExtractor(
        pretrain_results=pretrain_results,
        train_feature_names=train_features
    )
    
    # 4. 定义模型
    models = [
        TabPFNClassifier(device=device, model_path="tabpfn-v2-classifier.ckpt"),  # 保持不变
        CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, 
                           verbose=0, task_type="GPU" if device.type == 'cuda' else "CPU"),  # 保持不变
        xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.05, n_estimators=100, 
                         max_depth=5, random_state=42, tree_method='gpu_hist' if device.type == 'cuda' else 'hist'),  # 保持不变
        lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=50, learning_rate=0.05, 
                          n_estimators=100, min_split_gain=0.01, random_state=42,
                          device='gpu' if device.type == 'cuda' else 'cpu'),  # 保持不变
        SVC(probability=True, C=0.01, kernel='rbf', gamma='scale', random_state=42, 
            max_iter=10000)
    ]
    
    original_model_names = ["TRAIN", "CatBoost", "XGBoost", "LightGBM", "SVM"]
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    
    # 5. 训练与评估（仅SVM使用第一个代码的训练逻辑）
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
            
            # 仅SVM使用第一个代码的分层抽样，其他模型保持第二个代码的抽样方式
            if original_model_names[idx] == "SVM":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=i, stratify=y)  # 分层抽样
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=i)  # 普通抽样
                
            print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            # 获取特征：SVM使用标准化版本，其他模型使用原始版本
            X_train_enc, X_test_enc, X_train_enc_svm, X_test_enc_svm = feature_extractor.transform(X_train, X_test)
            
            # 根据模型类型选择特征
            if original_model_names[idx] == "SVM":
                curr_X_train = X_train_enc_svm
                curr_X_test = X_test_enc_svm
            else:
                curr_X_train = X_train_enc
                curr_X_test = X_test_enc
            
            start_time = time.time()
            
            # 仅SVM使用第一个代码的概率校准逻辑
            if original_model_names[idx] == "SVM":
                # 使用5折交叉验证进行概率校准
                calibrated_model = CalibratedClassifierCV(
                    estimator=model,
                    method='sigmoid',
                    cv=5
                )
                calibrated_model.fit(curr_X_train, y_train)
                current_model = calibrated_model
            else:
                # 其他模型保持第二个代码的训练方式
                model.fit(curr_X_train, y_train)
                current_model = model
                
            train_time = time.time() - start_time
            print(f"训练时间: {train_time:.2f}秒")
            
            y_pred = current_model.predict(curr_X_test)
            y_pred_proba = current_model.predict_proba(curr_X_test)[:, 1]
            
            # 仅SVM添加第一个代码的概率检查和异常处理
            if original_model_names[idx] == "SVM":
                print(f"SVM预测概率范围: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
                if np.allclose(y_pred_proba, 0) or np.allclose(y_pred_proba, 1):
                    print("警告：SVM预测概率异常接近0或1")
                
                # SVM的AUC异常处理
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    auc = 0.5 if len(np.unique(y_test)) < 2 else 0.0
                    print(f"迭代 {i+1} AUC计算异常，设置为{auc}")
            else:
                # 其他模型保持第二个代码的评估方式
                auc = roc_auc_score(y_test, y_pred_proba)
            
            accuracy = accuracy_score(y_test, y_pred)
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
    print("\n===== 多组学融合模型性能对比 =====")
    for idx, name in enumerate(original_model_names):
        print(f"{name}:")
        print(f"  平均准确率: {all_mean_accuracies[idx]:.3f} (±{all_std_accuracies[idx]:.3f})")
        print(f"  平均AUC: {all_mean_aucs[idx]:.3f} (±{all_std_aucs[idx]:.3f})")
        print("-" * 40)
    
    # 绘制ROC曲线
    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 21,
        'ytick.labelsize': 21,
        'legend.fontsize': 18,
    })
    
    plt.figure(figsize=(8, 6))
    for i in range(len(original_model_names)):
        plt.plot(mean_fpr, all_mean_tprs[i], lw=2,
                 label=f'{original_model_names[i]} AUROC: {all_mean_aucs[i]:.3f} (±{all_std_aucs[i]:.3f})',
                 color=colors[i])
        plt.fill_between(mean_fpr, all_mean_tprs[i] - all_std_tprs[i],
                         all_mean_tprs[i] + all_std_tprs[i], alpha=0.08, color=colors[i])
    
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Omics Fusion Models')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('multi_omics_fusion_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
    