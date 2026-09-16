import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNClassifier
import shap

# ===================== 基础配置 =====================
model_path = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ===================== 预训练模块 =====================
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
    def forward(self, x):
        return self.encoder(x)

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

def pretrain_feature_encoder(dataset_path, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    
    if 'group' in pretrain_feature_names:
        pretrain_feature_names.remove('group')
        X_pretrain = pretrain_data.drop(columns=['group']).values
    else:
        X_pretrain = pretrain_data.values
    
    imputer = SimpleImputer(strategy='mean')
    X_pretrain = imputer.fit_transform(X_pretrain)
    scaler = StandardScaler()
    X_pretrain = scaler.fit_transform(X_pretrain)
    
    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim=input_dim).to(device)
    decoder = FeatureDecoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    
    feature_var = np.var(X_pretrain, axis=0)
    var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
    mask_prob = 0.1 + 0.3 * (1 - var_norm)
    mask_prob_tensor = torch.FloatTensor(mask_prob).to(device)
    
    print(f"预训练 {dataset_path}：特征数{input_dim}，开始训练...")
    for epoch in range(epochs):
        total_loss = 0
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_pretrain)), batch_size=batch_size, shuffle=True)
        
        for batch in dataloader:
            x = batch[0].to(device)
            mask = torch.rand(x.shape).to(device) < mask_prob_tensor[None, :]
            x_masked = x.clone()
            x_masked[mask] = 0.0
            
            x_encoded = encoder(x_masked)
            x_reconstructed = decoder(x_encoded)
            loss = criterion(x_reconstructed[~mask], x[~mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(X_pretrain)
            print(f"Epoch {epoch+1}/{epochs} | 平均损失: {avg_loss:.4f}")
    
    return encoder, scaler, imputer, pretrain_feature_names

# ===================== 特征提取器 =====================
class MultiOmicsFeatureExtractor:
    def __init__(self, pretrain_results, train_feature_names):
        self.omics_extractors = []
        self.train_feature_names = train_feature_names
        self.encoded_to_original = []
        
        for encoder, scaler, imputer, pretrain_feats, omics_name in pretrain_results:
            self.omics_extractors.append({
                'encoder': encoder.to(device),
                'scaler': scaler,
                'imputer': imputer,
                'pretrain_feats': [f.strip().lower() for f in pretrain_feats],
                'name': omics_name,
                'pretrain_dim': len(pretrain_feats)
            })
        
        self.global_imputer = SimpleImputer(strategy='mean')
        self.global_scaler = StandardScaler()
        self._report_feature_matching()

    def _report_feature_matching(self):
        print("\n=== 特征匹配报告 ===")
        total_matched = set()
        for processor in self.omics_extractors:
            matched = [f for f in self.train_feature_names if f.strip().lower() in processor['pretrain_feats']]
            total_matched.update(matched)
            print(f"{processor['name']}: 匹配{len(matched)}/{len(self.train_feature_names)}个特征")
        print(f"总匹配特征: {len(total_matched)}/{len(self.train_feature_names)}个")

    def _get_matched_features(self, X, processor, is_train=True):
        matched_indices = [i for i, f in enumerate(self.train_feature_names) 
                          if f.strip().lower() in processor['pretrain_feats']]
        if not matched_indices:
            return np.array([]), matched_indices
        
        X_matched = X[:, matched_indices]
        if is_train:
            X_matched = processor['imputer'].fit_transform(X_matched)
            X_matched = processor['scaler'].fit_transform(X_matched)
        else:
            X_matched = processor['imputer'].transform(X_matched)
            X_matched = processor['scaler'].transform(X_matched)
        return X_matched, matched_indices

    def _get_unmatched_features(self, X, all_matched_indices, is_train=True):
        all_matched = set.union(*map(set, all_matched_indices))
        unmatched_indices = [i for i in range(X.shape[1]) if i not in all_matched]
        if not unmatched_indices:
            return np.array([]), unmatched_indices
        
        X_unmatched = X[:, unmatched_indices]
        if is_train:
            X_unmatched = self.global_imputer.fit_transform(X_unmatched)
            X_unmatched = self.global_scaler.fit_transform(X_unmatched)
        else:
            X_unmatched = self.global_imputer.transform(X_unmatched)
            X_unmatched = self.global_scaler.transform(X_unmatched)
        return X_unmatched, unmatched_indices

    def transform(self, X_train, X_test):
        print(f"\n=== 特征编码 ===")
        print(f"原始维度: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        train_encoded = []
        test_encoded = []
        all_matched_indices = []
        self.encoded_to_original = []
        
        for processor in self.omics_extractors:
            X_train_match, match_indices = self._get_matched_features(X_train, processor, is_train=True)
            X_test_match, _ = self._get_matched_features(X_test, processor, is_train=False)
            
            if X_train_match.size == 0:
                print(f"{processor['name']}: 无匹配特征，跳过")
                continue
            
            encoder = processor['encoder']
            if encoder.encoder[0].in_features != X_train_match.shape[1]:
                print(f"{processor['name']}: 适配编码器维度（{encoder.encoder[0].in_features}→{X_train_match.shape[1]}）")
                adapted_encoder = FeatureEncoder(input_dim=X_train_match.shape[1]).to(device)
                pretrained_dict = {k: v for k, v in encoder.state_dict().items() 
                                  if k in adapted_encoder.state_dict() and v.shape == adapted_encoder.state_dict()[k].shape}
                adapted_encoder.load_state_dict(pretrained_dict, strict=False)
                encoder = adapted_encoder
            
            encoder.eval()
            with torch.no_grad():
                X_train_enc = encoder(torch.FloatTensor(X_train_match).to(device)).cpu().numpy()
                X_test_enc = encoder(torch.FloatTensor(X_test_match).to(device)).cpu().numpy()
            
            match_names = [self.train_feature_names[i] for i in match_indices]
            enc_dim = X_train_enc.shape[1]
            if len(match_names) >= enc_dim:
                step = len(match_names) // enc_dim
                original_mapping = [match_names[i*step : (i+1)*step] for i in range(enc_dim)]
            else:
                original_mapping = [match_names[i % len(match_names):i % len(match_names)+1] for i in range(enc_dim)]
            self.encoded_to_original.extend(original_mapping)
            
            train_encoded.append(X_train_enc)
            test_encoded.append(X_test_enc)
            all_matched_indices.append(match_indices)
            print(f"{processor['name']}: 编码后维度{enc_dim}，映射原始特征{len(match_names)}个")
        
        X_train_unmatch, unmatch_indices = self._get_unmatched_features(X_train, all_matched_indices, is_train=True)
        X_test_unmatch, _ = self._get_unmatched_features(X_test, all_matched_indices, is_train=False)
        if X_train_unmatch.size > 0:
            train_encoded.append(X_train_unmatch)
            test_encoded.append(X_test_unmatch)
            unmatch_names = [self.train_feature_names[i] for i in unmatch_indices]
            self.encoded_to_original.extend([[name] for name in unmatch_names])
            print(f"未匹配特征: 维度{X_train_unmatch.shape[1]}")
        
        X_train_combined = np.hstack(train_encoded) if train_encoded else X_train
        X_test_combined = np.hstack(test_encoded) if test_encoded else X_test
        
        assert len(self.encoded_to_original) == X_train_combined.shape[1], \
            f"映射表长度与编码特征维度不匹配: {len(self.encoded_to_original)} vs {X_train_combined.shape[1]}"
        print(f"编码后维度: 训练集{X_train_combined.shape}, 测试集{X_test_combined.shape}")
        return X_train_combined, X_test_combined

# ===================== SHAP聚合工具 =====================
def aggregate_shap_to_original(shap_values, encoded_to_original, original_feature_names):
    n_samples = shap_values.shape[0]
    n_original = len(original_feature_names)
    original_shap = np.zeros((n_samples, n_original), dtype=np.float64)
    original_count = np.zeros(n_original, dtype=np.float64)

    for enc_idx, original_names in enumerate(encoded_to_original):
        curr_shap = shap_values[:, enc_idx]
        n_names = len(original_names)
        if n_names == 0 or np.all(curr_shap == 0):
            continue
        
        shap_per_name = curr_shap / n_names
        for name in original_names:
            if name not in original_feature_names:
                continue
            orig_idx = original_feature_names.index(name)
            original_shap[:, orig_idx] += shap_per_name
            original_count[orig_idx] += 1

    total_enc_shap = np.sum(shap_values)
    total_ori_shap = np.sum(original_shap)
    assert np.isclose(total_enc_shap, total_ori_shap, rtol=1e-3), \
        f"SHAP聚合前后总贡献不守恒！"
    
    return original_shap

# ===================== 主逻辑 =====================
def main():
    # 1. 多组学预训练
    print("="*50)
    print("=== 步骤1：多组学预训练 ===")
    print("="*50)
    pretrain_file_paths = [
        'Pre-关节联合.xlsx'
    ]
    pretrain_results = []
    for i, path in enumerate(pretrain_file_paths, 1):
        print(f"\n--- 预训练第{i}个数据集 ---")
        encoder, scaler, imputer, feats = pretrain_feature_encoder(dataset_path=path)
        pretrain_results.append( (encoder, scaler, imputer, feats, f"组学{i}") )

    # 2. 加载数据
    print("\n" + "="*50)
    print("=== 步骤2：加载数据 ===")
    print("="*50)
    data = pd.read_excel("DZCH影像.xlsx", header=0)
    X_original = data.iloc[:, 1:].values
    y = np.array([0 if 'N' in str(label) else 1 for label in data['group']])
    original_indices = np.arange(1, len(X_original) + 1)
    original_feature_names = data.columns[1:].tolist()
    print(f"样本数{len(X_original)}，特征数{len(original_feature_names)}")

    imputer = SimpleImputer(strategy='mean')
    X_original = imputer.fit_transform(X_original)
    scaler = StandardScaler()
    X_original = scaler.fit_transform(X_original)

    # 3. 初始化特征提取器
    feature_extractor = MultiOmicsFeatureExtractor(
        pretrain_results=pretrain_results,
        train_feature_names=original_feature_names
    )

    # 4. 10次迭代评估
    print("\n" + "="*50)
    print("=== 步骤3：模型评估（10次迭代） ===")
    print("="*50)
    accuracies = []
    aucs = []
    models = []
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    train_indices_list = []
    test_indices_list = []

    for i in range(10):
        print(f"\n--- 迭代{i+1}/10 ---")
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_original, y, original_indices, test_size=0.3, random_state=i
        )
        
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        train_indices_list.append(train_indices)
        test_indices_list.append(test_indices)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        clf = TabPFNClassifier(model_path=model_path, device=device)
        clf.fit(X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy())

        predictions = clf.predict(X_test_tensor.cpu().numpy())
        accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predictions)
        accuracies.append(accuracy)

        prediction_probabilities = clf.predict_proba(X_test_tensor.cpu().numpy())
        if len(prediction_probabilities.shape) == 1 or prediction_probabilities.shape[1] == 1:
            roc_auc = roc_auc_score(y_test_tensor.cpu().numpy(), prediction_probabilities)
        else:
            roc_auc = roc_auc_score(y_test_tensor.cpu().numpy(), prediction_probabilities[:, 1])
        aucs.append(roc_auc)

        models.append(clf)
        print(f"准确率: {accuracy:.3f}, AUC: {roc_auc:.3f}")

    # 5. 平均性能分析
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    accuracy_str = f"{mean_accuracy:.3f} ({mean_accuracy + std_accuracy:.3f}-{mean_accuracy - std_accuracy:.3f})"
    auc_str = f"{mean_auc:.3f} ({mean_auc + std_auc:.3f}-{mean_auc - std_auc:.3f})"
    print(f"\nMean Accuracy: {accuracy_str}")
    print(f"Mean AUC: {auc_str}")

    # 6. 保存最佳模型结果
    best_auc_index = np.argmax(aucs)
    best_model = models[best_auc_index]
    best_X_train = X_trains[best_auc_index]
    best_X_test = X_tests[best_auc_index]
    best_y_train = y_trains[best_auc_index]
    best_y_test = y_tests[best_auc_index]
    best_train_indices = train_indices_list[best_auc_index]
    best_test_indices = test_indices_list[best_auc_index]

    all_indices = np.hstack((best_train_indices, best_test_indices))
    all_y = np.hstack((best_y_train, best_y_test))
    group_index_df = pd.DataFrame({
        '分组信息': all_y,
        '原始编号': all_indices
    })
    group_index_df.to_excel('best_model_DZCH影像.xlsx', index=False)
    print("\n最佳模型分组信息如下：")
    print(group_index_df)

    # 7. SHAP分析
    print("\n" + "="*50)
    print("=== 步骤4：SHAP解释分析 ===")
    print("="*50)
    all_X_original = np.vstack((best_X_train, best_X_test))

    explainer = shap.Explainer(best_model.predict_proba, best_X_train)
    shap_values = explainer(all_X_original)
    shap_values_positive = shap_values[:, :, 1].values

    shap_df = pd.DataFrame(shap_values_positive, columns=original_feature_names)
    shap_df['group'] = all_y
    shap_df.to_excel('feature_shap_DZCH影像.xlsx', index=False)

    mean_shap = np.mean(shap_values_positive, axis=0)
    shap_rank_df = pd.DataFrame({
        'feature': original_feature_names,
        'mean_shap': mean_shap
    }).sort_values(by='mean_shap', key=lambda x: np.abs(x), ascending=False)
    shap_rank_df.to_excel('SHAP_Rank_DZCH影像.xlsx', index=False)

    print("\n所有分析流程已完成。")

if __name__ == "__main__":
    main()