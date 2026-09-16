import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

# ===================== 1. 文件及运行参数 =====================
PRETRAIN_FILE = "Pre-关节联合.xlsx"
DEVELOPMENT_FILE = "DZCH影像.xlsx"
MODEL_PATH = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
OUTPUT_FIGURE = "Phenotype: Imaging (IP vs DP).png"
N_REPEATS = 10
TEST_SIZE = 0.30
BASE_SEED = 42
ENCODER_SEED_OFFSET = 10000
MODEL_SEED_OFFSET = 20000

VALIDATION_CONFIGS = [
    {"name": "North", "file": "重新划分_Val1_北部区域.xlsx", "color": "#00A087"},
    {"name": "Central", "file": "重新划分_Val2_中部西部区域.xlsx", "color": "#E64B35"},
    {"name": "South", "file": "重新划分_Val4_南部区域.xlsx", "color": "#4DBBD5"}
]

# ===================== 2. 设置CUDA设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== 3. 确定性随机种子 =====================
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

set_all_seeds(BASE_SEED)

# ===================== 4. 定义特征编码器 =====================
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2))

    def forward(self, x):
        return self.encoder(x)

# ===================== 5. 定义特征解码器 =====================
class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(x)

# ===================== 6. 特征提取器 =====================
class UnifiedFeatureExtractor:
    def __init__(self, pretrain_encoder, pretrain_scaler, pretrain_imputer, pretrain_feature_names, train_feature_names):
        self.pretrain_encoder = pretrain_encoder.to(device)
        self.pretrain_scaler = pretrain_scaler
        self.pretrain_imputer = pretrain_imputer
        self.pretrain_feature_names = pretrain_feature_names
        self.train_feature_names = train_feature_names
        self.hidden_dim = 64
        self.train_to_pretrain = self._match_features()
        self.matched_indices = [i for i, idx in enumerate(self.train_to_pretrain) if idx is not None]
        self.unmatched_indices = [i for i, idx in enumerate(self.train_to_pretrain) if idx is None]
        self.train_imputer = SimpleImputer(strategy="mean")
        self.train_scaler = StandardScaler()

        print("\n特征匹配报告:")
        print(f"已匹配 {len(self.matched_indices)}/{len(train_feature_names)} 个特征")

        if self.unmatched_indices:
            print(f"未匹配 {len(self.unmatched_indices)} 个特征（示例）:")
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

        if self.matched_indices:
            if "encoder.0.weight" in pretrain_dict and "encoder.0.weight" in model_dict:
                pretrain_weight = pretrain_dict["encoder.0.weight"].clone()
                new_weight = model_dict["encoder.0.weight"].clone()

                for train_idx, pretrain_idx in enumerate(self.train_to_pretrain):
                    if pretrain_idx is not None and pretrain_idx < pretrain_weight.shape[1]:
                        new_weight[:, train_idx] = pretrain_weight[:, pretrain_idx]

                model_dict["encoder.0.weight"] = new_weight
                print(f"已迁移 {len(self.matched_indices)}/{len(self.train_to_pretrain)} 个匹配特征的第一层权重")

            for k in ["encoder.2.weight", "encoder.2.bias"]:
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

        X_train_combined = np.hstack([X_train_matched if X_train_matched.size > 0 else np.zeros((X_train.shape[0], 0)), X_train_unmatched if X_train_unmatched.size > 0 else np.zeros((X_train.shape[0], 0))])
        X_test_combined = np.hstack([X_test_matched if X_test_matched.size > 0 else np.zeros((X_test.shape[0], 0)), X_test_unmatched if X_test_unmatched.size > 0 else np.zeros((X_test.shape[0], 0))])

        encoder = self._adapt_encoder(X_train_combined.shape[1])
        encoder.eval()

        with torch.no_grad():
            X_train_enc = encoder(torch.FloatTensor(X_train_combined).to(device)).cpu().numpy()
            X_test_enc = encoder(torch.FloatTensor(X_test_combined).to(device)).cpu().numpy()

        print(f"编码后维度: 训练集{X_train_enc.shape}, 测试集{X_test_enc.shape}")
        return X_train_enc, X_test_enc

# ===================== 7. 掩码自监督预训练 =====================
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    set_all_seeds(BASE_SEED)
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)
    pretrain_feature_names.remove("group")

    X_pretrain = pretrain_data.drop(columns=["group"]).values
    imputer = SimpleImputer(strategy="mean")
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return encoder, scaler, imputer, pretrain_feature_names

# ===================== 8. 数据集检查 =====================
def prepare_dataset(data, train_features, dataset_name):
    if "group" not in data.columns:
        raise ValueError(f"{dataset_name}缺少group列")

    missing_features = [col for col in train_features if col not in data.columns]

    if missing_features:
        raise ValueError(f"{dataset_name}缺少训练特征：{missing_features}")

    y = data["group"].astype(str).str.strip().str.upper().map({"N": 0, "T": 1})

    if y.isna().any():
        invalid_values = data.loc[y.isna(), "group"].drop_duplicates().tolist()
        raise ValueError(f"{dataset_name}存在非N/T标签：{invalid_values}")

    if y.nunique() < 2:
        raise ValueError(f"{dataset_name}只有一个类别，不能计算AUROC")

    X = data[train_features].values
    return X, y.astype(int).values

# ===================== 9. 单个外部验证集运行 =====================
def evaluate_external_validation(cohort_name, cohort_color, X_internal, y_internal, X_external, y_external, feature_extractor, mean_fpr):
    accuracies = []
    aucs = []
    fprs = []
    tprs = []

    print(f"\n===== TRAIN模型：{cohort_name}独立外部验证 =====")

    for i in range(N_REPEATS):
        split_seed = i
        encoder_seed = BASE_SEED + ENCODER_SEED_OFFSET + i
        model_seed = BASE_SEED + MODEL_SEED_OFFSET + i

        print(f"\n{cohort_name}迭代 {i + 1}/{N_REPEATS}")
        print(f"数据划分种子={split_seed}，编码器种子={encoder_seed}，模型种子={model_seed}")

        X_train, _, y_train, _ = train_test_split(X_internal, y_internal, test_size=TEST_SIZE, random_state=split_seed)
        print(f"训练集: {X_train.shape}, {cohort_name}验证集: {X_external.shape}")

        set_all_seeds(encoder_seed)
        X_train_enc, X_external_enc = feature_extractor.transform(X_train, X_external)

        set_all_seeds(model_seed)
        model = TabPFNClassifier(device=device, model_path=MODEL_PATH)

        start_time = time.time()
        model.fit(X_train_enc, y_train)
        print(f"训练时间: {time.time() - start_time:.2f}秒")

        y_pred = model.predict(X_external_enc)
        y_pred_proba = model.predict_proba(X_external_enc)[:, 1]

        accuracy = accuracy_score(y_external, y_pred)
        auc_value = roc_auc_score(y_external, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_external, y_pred_proba)

        accuracies.append(accuracy)
        aucs.append(auc_value)
        fprs.append(fpr)
        tprs.append(tpr)

        print(f"{cohort_name}迭代 {i + 1} 性能: 准确率={accuracy:.3f}, AUROC={auc_value:.3f}")

    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)]

    return {
        "name": cohort_name,
        "color": cohort_color,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "mean_tpr": np.mean(interp_tprs, axis=0),
        "std_tpr": np.std(interp_tprs, axis=0)
    }

# ===================== 10. 主函数 =====================
def main():
    required_files = [PRETRAIN_FILE, DEVELOPMENT_FILE] + [config["file"] for config in VALIDATION_CONFIGS]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件：{file_path}")

    validation_names = [config["name"] for config in VALIDATION_CONFIGS]

    if len(validation_names) != len(set(validation_names)):
        raise ValueError("VALIDATION_CONFIGS中存在重复的验证集名称")

    print("开始预训练...")
    encoder, scaler, imputer, pretrain_features = pretrain_feature_encoder(dataset_path=PRETRAIN_FILE, epochs=50)

    print("\n加载训练数据...")
    train_data = pd.read_excel(DEVELOPMENT_FILE)
    train_features = [col for col in train_data.columns if col != "group"]
    X_internal, y_internal = prepare_dataset(train_data, train_features, "DZCH开发集")
    print(f"DZCH开发集: {X_internal.shape}, N={(y_internal == 0).sum()}, T={(y_internal == 1).sum()}")
    print(f"每次使用DZCH开发集的{(1 - TEST_SIZE) * 100:.0f}%训练，剩余{TEST_SIZE * 100:.0f}%不参与外部验证训练")

    validation_datasets = []

    for config in VALIDATION_CONFIGS:
        print(f"\n加载{config['name']}外部验证数据...")
        val_data = pd.read_excel(config["file"])
        X_external, y_external = prepare_dataset(val_data, train_features, config["name"])
        validation_datasets.append({"name": config["name"], "color": config["color"], "X": X_external, "y": y_external})
        print(f"{config['name']}验证集: {X_external.shape}, N={(y_external == 0).sum()}, T={(y_external == 1).sum()}")

    feature_extractor = UnifiedFeatureExtractor(encoder, scaler, imputer, pretrain_features, train_features)
    mean_fpr = np.linspace(0, 1, 100)
    all_results = []

    for dataset in validation_datasets:
        external_result = evaluate_external_validation(dataset["name"], dataset["color"], X_internal, y_internal, dataset["X"], dataset["y"], feature_extractor, mean_fpr)
        all_results.append(external_result)

    print("\n===== TRAIN模型三个独立外部验证集AUROC结果 =====")

    for result in all_results:
        print(f"{result['name']}:")
        print(f"  平均准确率: {result['mean_accuracy']:.3f} (±{result['std_accuracy']:.3f})")
        print(f"  平均AUROC: {result['mean_auc']:.3f} (±{result['std_auc']:.3f})")
        print("-" * 40)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 22, "axes.titlesize": 22, "axes.labelsize": 22, "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 16})
    plt.figure(figsize=(8, 6))

    for result in all_results:
        lower_tpr = np.clip(result["mean_tpr"] - result["std_tpr"], 0, 1)
        upper_tpr = np.clip(result["mean_tpr"] + result["std_tpr"], 0, 1)
        plt.plot(mean_fpr, result["mean_tpr"], lw=2, label=f"{result['name']} AUROC: {result['mean_auc']:.3f} (±{result['std_auc']:.3f})", color=result["color"])
        plt.fill_between(mean_fpr, lower_tpr, upper_tpr, alpha=0.08, color=result["color"])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation: Phenotype: Imaging (IP vs DP)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()