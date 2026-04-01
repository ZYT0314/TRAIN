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

# 定义特征编码器
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

# 定义特征解码器用于自监督预训练
class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)

# 特征提取器（统一处理特征匹配和维度转换）
class UnifiedFeatureExtractor:
    def __init__(self, pretrain_encoder, pretrain_scaler, pretrain_imputer, 
                 pretrain_feature_names, train_feature_names):
        self.pretrain_encoder = pretrain_encoder.to(device)
        self.pretrain_scaler = pretrain_scaler
        self.pretrain_imputer = pretrain_imputer
        self.pretrain_feature_names = pretrain_feature_names
        self.train_feature_names = train_feature_names
        self.hidden_dim = 64
        
        self.train_feature_indices, self.mapping_report = self._match_features()
        self.train_imputer = SimpleImputer(strategy='mean')
        self.train_scaler = StandardScaler()
        
        print("\n特征匹配报告:")
        matched_count = sum(1 for idx in self.train_feature_indices if idx is not None)
        print(f"已匹配 {matched_count}/{len(train_feature_names)} 个特征")
    
    def _match_features(self):
        pretrain_names = [name.strip().lower() for name in self.pretrain_feature_names]
        train_names = [name.strip().lower() for name in self.train_feature_names]
        
        indices = []
        mapping = {}
        
        for train_idx, train_feat in enumerate(train_names):
            if train_feat in pretrain_names:
                pretrain_idx = pretrain_names.index(train_feat)
                indices.append(pretrain_idx)
                mapping[self.train_feature_names[train_idx]] = pretrain_idx
            else:
                indices.append(None)
                mapping[self.train_feature_names[train_idx]] = None
        
        unmatched = [i for i, idx in enumerate(indices) if idx is None]
        if unmatched:
            print(f"⚠️ 警告: {len(unmatched)}/{len(train_names)} 个特征未匹配:")
            for i in unmatched[:5]:  # 限制显示数量，避免输出过长
                print(f"  - '{self.train_feature_names[i]}'")
        
        return indices, mapping
    
    def _adapt_encoder(self, train_feature_dim):
        new_encoder = FeatureEncoder(input_dim=train_feature_dim, hidden_dim=self.hidden_dim).to(device)
        
        pretrain_dict = self.pretrain_encoder.state_dict()
        model_dict = new_encoder.state_dict()
        
        # 改进权重迁移逻辑，只迁移形状匹配的层
        transferable = {k: v for k, v in pretrain_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(transferable)
        new_encoder.load_state_dict(model_dict)
        
        return new_encoder
    
    def transform(self, X_train, X_test):
        print(f"\n特征提取: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 处理缺失值
        self.train_imputer.fit(X_train)
        X_train_imputed = self.train_imputer.transform(X_train)
        X_test_imputed = self.train_imputer.transform(X_test)
        
        # 标准化
        self.train_scaler.fit(X_train_imputed)
        X_train_scaled = self.train_scaler.transform(X_train_imputed)
        X_test_scaled = self.train_scaler.transform(X_test_imputed)
        
        # 适配编码器
        train_dim = X_train.shape[1]
        encoder = self._adapt_encoder(train_dim)
        
        # 特征编码
        encoder.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
            X_train_encoded = encoder(X_train_tensor).cpu().numpy()
            
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            X_test_encoded = encoder(X_test_tensor).cpu().numpy()
        
        print(f"编码后维度: 训练集{X_train_encoded.shape}, 测试集{X_test_encoded.shape}")
        return X_train_encoded, X_test_encoded

# 掩码自监督预训练函数
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    """通过掩码自编码器预训练特征表示"""
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    pretrain_feature_names.remove('group')  # 移除标签列
    
    X_pretrain = pretrain_data.drop(columns=['group']).values
    
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
    
    # 将掩码比例转换为张量，避免类型不匹配
    mask_ratio_tensor = torch.tensor(mask_ratio, device=device)
    
    print(f"开始预训练 {input_dim} 维特征编码器...")
    for epoch in range(epochs):
        total_loss = 0
        # 数据加载
        dataloader = DataLoader(
            TensorDataset(torch.FloatTensor(X_pretrain)),
            batch_size=batch_size, 
            shuffle=True
        )
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # 生成掩码（修复类型不匹配问题）
            mask = torch.rand(x.shape, device=device) < mask_ratio_tensor
            x_masked = x.clone()
            x_masked[mask] = 0.0  # 掩码位置置零
            
            # 编码解码过程
            encoded = encoder(x_masked)
            decoded = decoder(encoded)
            loss = criterion(decoded, x)  # 计算重建损失
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return encoder, scaler, imputer, pretrain_feature_names

# 主函数
def main():
    # 预训练
    print("开始预训练...")
    encoder, scaler, imputer, pretrain_features = pretrain_feature_encoder(
        dataset_path='Pre-RADB204样本.xlsx',  # 更新为你的预训练数据文件
        epochs=50
    )
    
    # 加载训练数据
    print("\n加载训练数据...")
    train_data = pd.read_excel('RA_DEPDB133fc1.2.xlsx')  # 更新为你的训练数据文件
    train_features = [col for col in train_data.columns if col != 'group']
    y = train_data['group'].map({'N': 0, 'T': 1}).values  # 标签映射
    X = train_data[train_features].values
    
    # 创建特征提取器
    feature_extractor = UnifiedFeatureExtractor(
        encoder, scaler, imputer, pretrain_features, train_features
    )
    
    # 定义模型列表
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
    
    model_names = ["TRAIN", "CatBoost", "XGBoost", "LightGBM", "SVM"]
    colors = ['blue', 'green', 'magenta', 'cyan', 'orange']  # 保持颜色映射
    
    # 存储性能指标
    all_mean_auprcs = []
    all_std_auprcs = []
    all_mean_precisions = []
    all_std_precisions = []
    all_mean_accuracies = []
    all_std_accuracies = []
    
    # 训练每个模型
    for idx, model in enumerate(models):
        accuracies = []
        auprcs = []
        precisions_list = []
        recalls_list = []
        
        print(f"\n开始训练 {model_names[idx]} 模型...")
        
        # 十次迭代
        for i in range(10):
            print(f"迭代 {i+1}/10")
            
            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=i
            )
            
            print(f"训练集{X_train.shape}, 测试集{X_test.shape}")
            
            # 特征编码
            X_train_enc, X_test_enc = feature_extractor.transform(X_train, X_test)
            
            # 训练模型
            start_time = time.time()
            model.fit(X_train_enc, y_train)
            train_time = time.time() - start_time
            print(f"训练时间: {train_time:.2f}秒")
            
            # 预测与评估
            y_pred = model.predict(X_test_enc)
            y_pred_proba = model.predict_proba(X_test_enc)[:, 1]
            
            # 计算准确度
            accuracy = accuracy_score(y_test, y_pred)
            
            # 计算PRC曲线和AUPRC
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = auc(recall, precision)
            
            accuracies.append(accuracy)
            auprcs.append(auprc)
            precisions_list.append(precision)
            recalls_list.append(recall)
        
        # 计算平均指标
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)
        
        # 保存统计量
        all_mean_accuracies.append(mean_accuracy)
        all_std_accuracies.append(std_accuracy)
        all_mean_auprcs.append(mean_auprc)
        all_std_auprcs.append(std_auprc)
        
        # 对PRC曲线进行插值
        mean_recall = np.linspace(0, 1, 100)
        interp_precisions = []
        
        for precision, recall in zip(precisions_list, recalls_list):
            # 排序确保插值正确性
            idx_sort = np.argsort(recall)
            sorted_recall = recall[idx_sort]
            sorted_precision = precision[idx_sort]
            
            # 插值到统一网格
            interp_precision = np.interp(mean_recall, sorted_recall, sorted_precision)
            interp_precisions.append(interp_precision)
        
        mean_precision = np.mean(interp_precisions, axis=0)
        std_precision = np.std(interp_precisions, axis=0)
        
        all_mean_precisions.append(mean_precision)
        all_std_precisions.append(std_precision)
        
        # 输出当前模型AUPRC结果（仅保留AUPRC打印，删除Accuracy打印）
        auprc_str = f"{mean_auprc:.3f} (±{std_auprc:.3f})"
        print(f"{model_names[idx]} Mean AUPRC: {auprc_str}")
    
    # 按期望顺序打印结果
    desired_print_order = ["LightGBM", "TRAIN", "CatBoost", "XGBoost", "SVM"]
    name_to_desired_idx = {name: i for i, name in enumerate(desired_print_order)}
    desired_to_original = {desired_idx: original_idx 
                          for original_idx, name in enumerate(model_names)
                          for desired_idx, desired_name in enumerate(desired_print_order)
                          if name == desired_name}
    
    print("\n===== 模型AUPRC性能对比 =====")
    for desired_idx in range(len(desired_print_order)):
        original_idx = desired_to_original[desired_idx]
        name = model_names[original_idx]
        
        mean_auprc = all_mean_auprcs[original_idx]
        std_auprc = all_std_auprcs[original_idx]
        
        auprc_str = f"{mean_auprc:.3f} (±{std_auprc:.3f})"
        print(f"{name} Mean AUPRC: {auprc_str}")
        print("-" * 40)
    
    # 绘制PRC曲线（应用指定的字体设置）
    plt.rcParams.update({
        'font.size': 22,          # 全局字体大小
        'axes.titlesize': 22,     # 标题字体大小
        'axes.labelsize': 22,     # 坐标轴标签字体大小
        'xtick.labelsize': 21,    # x轴刻度字体大小
        'ytick.labelsize': 21,    # y轴刻度字体大小
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
    plt.title('PRC Curve for Proteomics Models')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('comparison_prc_curves.png', dpi=300, bbox_inches='tight')  # 确保图例完整保存
    plt.show()

if __name__ == "__main__":
    main()