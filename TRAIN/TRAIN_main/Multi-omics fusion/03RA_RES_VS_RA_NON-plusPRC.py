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
        
        # 存储每个组学的预训练信息（仅保留编码器和特征名用于匹配）
        for encoder, _, _, pretrain_feats, omics_name in pretrain_results:
            extractor = {
                'encoder': encoder.to(device),
                'pretrain_feats': [f.strip().lower() for f in pretrain_feats],  # 预训练特征名（小写标准化）
                'name': omics_name,
                'pretrain_input_dim': encoder.encoder[0].in_features  # 记录预训练时的输入维度
            }
            self.omics_extractors.append(extractor)
        
        # 为每个组学的匹配特征创建专用处理器（解决维度不匹配问题）
        self.group_imputers = {}  # 格式: {组学名称: 处理器}
        self.group_scalers = {}   # 格式: {组学名称: 处理器}
        
        # 未匹配特征的公共处理器（非预训练原始处理）
        self.common_imputer = SimpleImputer(strategy='mean')
        self.common_scaler = StandardScaler()
        
        # 特征匹配报告
        self._report_feature_matching()

    def _report_feature_matching(self):
        """报告每个组学与融合数据的特征匹配情况"""
        print("\n多组学特征匹配报告:")
        for extractor in self.omics_extractors:
            matched_feats = [f for f in self.train_feature_names 
                          if f.strip().lower() in extractor['pretrain_feats']]
            print(f"{extractor['name']}: 匹配特征 {len(matched_feats)} 个 (融合数据总特征: {len(self.train_feature_names)})")
            if matched_feats:
                print(f"  匹配特征示例: {matched_feats[:3]}..." if len(matched_feats) > 3 else f"  匹配特征: {matched_feats}")

    def _get_matched_features(self, X, extractor, is_train=True):
        """
        仅对匹配特征进行迁移学习：
        1. 提取融合数据中与预训练匹配的特征
        2. 使用为该组学匹配特征专门训练的处理器（避免维度不匹配）
        3. 通过预训练编码器提取特征（迁移学习）
        """
        # 1. 找到匹配特征的索引
        matched_indices = [i for i, f in enumerate(self.train_feature_names)
                          if f.strip().lower() in extractor['pretrain_feats']]
        if not matched_indices:  # 无匹配特征则返回空
            return np.array([]), matched_indices
        
        # 2. 提取匹配特征矩阵
        X_matched = X[:, matched_indices]
        
        # 3. 为匹配特征创建/使用专用处理器（解决维度不匹配问题）
        group_key = extractor['name']
        if group_key not in self.group_imputers:
            self.group_imputers[group_key] = SimpleImputer(strategy='mean')
            self.group_scalers[group_key] = StandardScaler()
        
        # 训练集拟合处理器，测试集直接转换
        if is_train:
            X_matched = self.group_imputers[group_key].fit_transform(X_matched)
            X_matched = self.group_scalers[group_key].fit_transform(X_matched)
        else:
            X_matched = self.group_imputers[group_key].transform(X_matched)
            X_matched = self.group_scalers[group_key].transform(X_matched)
            
        return X_matched, matched_indices

    def _get_unmatched_features(self, X, matched_indices_all, is_train=True):
        """
        未匹配特征使用非预训练原始处理：
        - 不使用任何预训练知识，直接在融合数据上拟合处理器
        """
        # 合并所有匹配索引，找到未匹配特征
        all_matched_indices = set()
        for indices in matched_indices_all:
            all_matched_indices.update(indices)
        unmatched_indices = [i for i in range(X.shape[1]) if i not in all_matched_indices]
        
        if not unmatched_indices:
            return np.array([])
            
        # 原始处理：直接在融合数据上训练处理器
        X_unmatched = X[:, unmatched_indices]
        if is_train:
            X_unmatched = self.common_imputer.fit_transform(X_unmatched)
            X_unmatched = self.common_scaler.fit_transform(X_unmatched)
        else:
            X_unmatched = self.common_imputer.transform(X_unmatched)
            X_unmatched = self.common_scaler.transform(X_unmatched)
        return X_unmatched

    def transform(self, X_train, X_test):
        """
        特征融合逻辑：
        1. 对每个组学的匹配特征：迁移学习（预训练编码器）
        2. 对未匹配特征：原始处理（不使用预训练）
        3. 拼接所有特征作为最终输入
        """
        print(f"\n多组学特征提取: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        train_encoded_features = []  # 匹配特征的编码结果（迁移学习）
        test_encoded_features = []
        matched_indices_all = []     # 所有组学的匹配特征索引
        
        # 处理每个组学的匹配特征
        for extractor in self.omics_extractors:
            # 提取并预处理匹配特征
            X_train_matched, matched_indices = self._get_matched_features(
                X_train, extractor, is_train=True)
            X_test_matched, _ = self._get_matched_features(
                X_test, extractor, is_train=False)
            
            matched_indices_all.append(matched_indices)
            
            if X_train_matched.size == 0:  # 无匹配特征则跳过
                print(f"{extractor['name']}: 无匹配特征，不进行迁移学习")
                continue
                
            # 适配编码器输入维度（仅调整第一层，保留其他层预训练知识）
            current_encoder = extractor['encoder']
            current_input_dim = X_train_matched.shape[1]
            pretrain_input_dim = extractor['pretrain_input_dim']
            
            if current_input_dim != pretrain_input_dim:
                print(f"{extractor['name']}: 匹配特征维度 {current_input_dim} ≠ 预训练维度 {pretrain_input_dim}，适配编码器")
                # 创建新编码器，仅复用除第一层外的预训练权重
                adapted_encoder = FeatureEncoder(input_dim=current_input_dim).to(device)
                pretrained_dict = current_encoder.state_dict()
                adapted_dict = adapted_encoder.state_dict()
                
                # 只保留可复用的权重（排除第一层）
                reusable_keys = [k for k in pretrained_dict.keys() if not k.startswith('encoder.0.')]
                for key in reusable_keys:
                    adapted_dict[key] = pretrained_dict[key]
                
                adapted_encoder.load_state_dict(adapted_dict)
                current_encoder = adapted_encoder
            
            # 迁移学习：使用预训练编码器提取特征
            current_encoder.eval()
            with torch.no_grad():
                X_train_enc = current_encoder(torch.FloatTensor(X_train_matched).to(device)).cpu().numpy()
                X_test_enc = current_encoder(torch.FloatTensor(X_test_matched).to(device)).cpu().numpy()
            
            train_encoded_features.append(X_train_enc)
            test_encoded_features.append(X_test_enc)
            print(f"{extractor['name']}: 迁移学习后特征维度 {X_train_enc.shape[1]}")
        
        # 处理未匹配特征（非预训练原始处理）
        X_train_unmatched = self._get_unmatched_features(X_train, matched_indices_all, is_train=True)
        X_test_unmatched = self._get_unmatched_features(X_test, matched_indices_all, is_train=False)
        
        # 合并所有特征
        all_train_features = train_encoded_features
        all_test_features = test_encoded_features
        
        if X_train_unmatched.size > 0:
            all_train_features.append(X_train_unmatched)
            all_test_features.append(X_test_unmatched)
            print(f"未匹配特征: 原始处理后维度 {X_train_unmatched.shape[1]}")
        
        # 最终特征拼接
        X_train_combined = np.hstack(all_train_features) if all_train_features else X_train
        X_test_combined = np.hstack(all_test_features) if all_test_features else X_test
        
        print(f"最终融合特征: 训练集{X_train_combined.shape}, 测试集{X_test_combined.shape}")
        return X_train_combined, X_test_combined

# 自监督预训练函数（仅用于"热身"，学习数据分布特征）
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    
    if 'group' in pretrain_feature_names:
        pretrain_feature_names.remove('group')
        X_pretrain = pretrain_data.drop(columns=['group']).values
    else:
        X_pretrain = pretrain_data.values
    
    # 预训练数据预处理（仅用于编码器训练，不用于后续迁移）
    imputer = SimpleImputer(strategy='mean')
    X_pretrain = imputer.fit_transform(X_pretrain)
    scaler = StandardScaler()
    X_pretrain = scaler.fit_transform(X_pretrain)
    
    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim).to(device)
    decoder = FeatureDecoder(input_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    # 智能掩码策略（增强预训练效果）
    feature_var = np.var(X_pretrain, axis=0)
    var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
    mask_prob = 0.1 + 0.3 * (1 - var_norm)  # 方差低的特征更容易被掩码
    mask_prob_tensor = torch.FloatTensor(mask_prob).to(device)
    print(f"预训练特征数: {input_dim}, 采用智能掩码策略")
    
    # 预训练循环（仅作为"热身"，学习数据分布）
    print("开始预训练编码器（热身）...")
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
    
    # 仅返回编码器和特征名（预处理处理器不用于后续迁移）
    return encoder, None, None, pretrain_feature_names

# 主函数
def main():
    # 1. 多组学预训练（仅作为热身）
    print("===== 多组学预训练阶段（热身） =====")
    pretrain_file_paths = [
        'Pre-LCRA_VS_RADEP39.xlsx',   # 组学1
        'Pre-RABC_RAresponse67.xlsx',    # 组学2
        'Pre-RADB204样本.xlsx'          # 组学3
    ]
    pretrain_results = []
    
    for i, file_path in enumerate(pretrain_file_paths, 1):
        print(f"\n----- 开始第 {i} 个组学预训练（文件：{file_path}）-----")
        encoder, _, _, pretrain_features = pretrain_feature_encoder(
            dataset_path=file_path,
            epochs=50
        )
        pretrain_results.append( (encoder, None, None, pretrain_features, f"组学{i}") )
        print(f"----- 第 {i} 个组学预训练完成 -----")
    
    # 2. 加载融合分析数据（核心训练数据）
    print("\n===== 加载融合分析数据 =====")
    train_data = pd.read_excel('临床转录蛋白应答不应答合并新.xlsx')
    train_features = [col for col in train_data.columns if col != 'group']
    
    if 'group' not in train_data.columns:
        raise ValueError("融合数据必须包含'group'列作为标签")
    y = train_data['group'].map({'N': 0, 'T': 1}).values
    X = train_data[train_features].values
    
    # 3. 初始化特征提取器（核心逻辑：匹配特征迁移学习，未匹配特征原始处理）
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
        lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=50, learning_rate=0.05, 
                          n_estimators=100, min_split_gain=0.01, random_state=42,
                          device='gpu' if device.type == 'cuda' else 'cpu'),
        SVC(probability=True, C=0.001, kernel='linear', gamma='scale', random_state=42)
    ]
    
    original_model_names = ["TRAIN", "CatBoost", "XGBoost", "LightGBM", "SVM"]
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
    
    # 5. 训练与评估（核心修改：ROC→PRC）
    all_mean_auprcs = []    # 替换原all_mean_aucs
    all_std_auprcs = []     # 替换原all_std    all_mean_auprcs = []    # 存储各模型平均AUPRC
    all_std_auprcs = []     # 存储各模型AUPRC标准差
    all_mean_precisions = []# 存储各模型平均Precision（用于PRC曲线）
    all_std_precisions = [] # 存储各模型Precision标准差
    mean_recall = np.linspace(0, 1, 100)  # 与第二个代码一致：用于PRC曲线插值的统一Recall网格
    
    for idx, model in enumerate(models):
        accuracies = []
        auprcs = []
        precisions_list = []  # 存储每次迭代的Precision（与第二个代码一致）
        recalls_list = []     # 存储每次迭代的Recall（与第二个代码一致）
        
        print(f"\n===== 训练 {original_model_names[idx]} 模型 =====")
        
        for i in range(10):
            print(f"\n迭代 {i+1}/10")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=i)
            print(f"原始特征: 训练集{X_train.shape}, 测试集{X_test.shape}")
            
            # 提取融合特征（保持原多组学逻辑不变）
            X_train_enc, X_test_enc = feature_extractor.transform(X_train, X_test)
            
            # 训练模型
            start_time = time.time()
            model.fit(X_train_enc, y_train)
            print(f"训练时间: {time.time() - start_time:.2f}秒")
            
            # 评估（严格对齐第二个代码的PRC计算逻辑）
            y_pred = model.predict(X_test_enc)
            y_pred_proba = model.predict_proba(X_test_enc)[:, 1]
            
            # 计算准确率（保留原指标）
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            # 计算PRC曲线和AUPRC（与第二个代码完全一致）
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = auc(recall, precision)
            auprcs.append(auprc)
            
            print(f"迭代 {i+1} 性能: 准确率={accuracy:.3f}, AUPRC={auprc:.3f}")
            
            # 保存Precision和Recall（与第二个代码一致）
            precisions_list.append(precision)
            recalls_list.append(recall)
        
        # 计算当前模型的平均指标（与第二个代码逻辑一致）
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)
        
        # 存储指标
        all_mean_auprcs.append(mean_auprc)
        all_std_auprcs.append(std_auprc)
        
        # PRC曲线插值（严格复现第二个代码的插值逻辑）
        interp_precisions = []
        for precision, recall in zip(precisions_list, recalls_list):
            # 排序确保插值正确性（与第二个代码完全一致）
            idx_sort = np.argsort(recall)
            sorted_recall = recall[idx_sort]
            sorted_precision = precision[idx_sort]
            
            # 插值到统一网格（与第二个代码完全一致）
            interp_precision = np.interp(mean_recall, sorted_recall, sorted_precision)
            interp_precisions.append(interp_precision)
        
        # 计算平均Precision和标准差（与第二个代码一致）
        mean_precision = np.mean(interp_precisions, axis=0)
        std_precision = np.std(interp_precisions, axis=0)
        all_mean_precisions.append(mean_precision)
        all_std_precisions.append(std_precision)
    
    # 打印性能对比（保持原格式，仅替换AUC为AUPRC）
    print("\n===== 多组学融合模型性能对比 =====")
    for idx, name in enumerate(original_model_names):
        print(f"{name}:")
        print(f"  平均AUPRC: {all_mean_auprcs[idx]:.3f} (±{all_std_auprcs[idx]:.3f})")
        print("-" * 40)
    
    # 绘制PRC曲线（严格按照第二个代码的绘图参数和格式）
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
        # 绘制平均PRC曲线（与第二个代码一致：线宽、标签格式、颜色映射）
        plt.plot(mean_recall, all_mean_precisions[i], lw=2,
                 label=f'{original_model_names[i]} AUPRC: {all_mean_auprcs[i]:.3f} (±{all_std_auprcs[i]:.3f})',
                 color=colors[i])
        # 绘制标准差填充区域（与第二个代码一致：透明度、颜色）
        plt.fill_between(mean_recall, all_mean_precisions[i] - all_std_precisions[i],
                         all_mean_precisions[i] + all_std_precisions[i], alpha=0.08, color=colors[i])
    
    # 严格对齐第二个代码的坐标轴和标题设置（无随机猜测线）
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRC Curve for Multi-Omics Fusion Models')  # 标题适配多组学场景
    plt.legend(loc="lower right")  # 恢复与第二个代码一致的图例位置
    plt.tight_layout()
    plt.savefig('multi_omics_fusion_prc_curves.png', dpi=300, bbox_inches='tight')  # 保存路径适配原代码
    plt.show()

if __name__ == "__main__":
    main()