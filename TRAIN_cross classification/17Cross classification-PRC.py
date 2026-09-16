import os
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
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

# ===================== 1. 文件及运行参数 =====================
DEVELOPMENT_FILE = "DZCH疾病分类全部数据.xlsx"
MODEL_PATH = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
OUTPUT_FIGURE = "Development_Clinical_Pairwise_PRC.png"
N_REPEATS = 10
TEST_SIZE = 0.30
BASE_SEED = 42

# DZCH疾病分类全部数据.xlsx前三个Sheet按顺序分别对应三个任务
TASK_CONFIGS = [
    {"name": "RA vs SLE", "sheet_index": 0, "pretrain_file": "Pre-multicalss.xlsx", "color": "#00A087"},
    {"name": "RA vs AS", "sheet_index": 1, "pretrain_file": "Pre-multicalss.xlsx", "color": "#E64B35"},
    {"name": "SLE vs AS", "sheet_index": 2, "pretrain_file": "Pre-multicalss.xlsx", "color": "#4DBBD5"}
]

# ===================== 2. 设置CUDA设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== 3. 设置随机种子 =====================
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(BASE_SEED)

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

# ===================== 7. 自监督预训练 =====================
def pretrain_feature_encoder(dataset_path, mask_ratio=0.2, epochs=50, batch_size=64, lr=1e-3):
    pretrain_data = pd.read_excel(dataset_path)
    pretrain_feature_names = list(pretrain_data.columns)

    if "group" in pretrain_feature_names:
        pretrain_feature_names.remove("group")
        X_pretrain = pretrain_data.drop(columns=["group"]).values
    else:
        X_pretrain = pretrain_data.values

    imputer = SimpleImputer(strategy="mean")
    X_pretrain = imputer.fit_transform(X_pretrain)
    scaler = StandardScaler()
    X_pretrain = scaler.fit_transform(X_pretrain)

    input_dim = X_pretrain.shape[1]
    encoder = FeatureEncoder(input_dim).to(device)
    decoder = FeatureDecoder(input_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()

    feature_var = np.var(X_pretrain, axis=0)
    var_norm = (feature_var - feature_var.min()) / (feature_var.max() - feature_var.min() + 1e-8)
    mask_prob = 0.1 + 0.3 * (1 - var_norm)
    mask_prob_tensor = torch.FloatTensor(mask_prob).to(device)

    print(f"预训练特征数: {input_dim}, 采用智能掩码策略")
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
            print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")

    return encoder, scaler, imputer, pretrain_feature_names

# ===================== 8. 检查并准备二分类数据 =====================
def prepare_dataset(data, task_name):
    if "group" not in data.columns:
        raise ValueError(f"{task_name}对应Sheet缺少group列")

    train_features = [col for col in data.columns if col != "group"]
    group = data["group"].astype(str).str.strip().str.upper()
    y = group.map({"N": 0, "T": 1})

    if y.isna().any():
        invalid_values = data.loc[y.isna(), "group"].drop_duplicates().tolist()
        raise ValueError(f"{task_name}存在非N/T标签：{invalid_values}")

    if y.nunique() != 2:
        raise ValueError(f"{task_name}必须同时包含N和T两个类别")

    X = data[train_features].values
    return X, y.astype(int).values, train_features

# ===================== 9. 单个疾病对独立运行 =====================
def run_single_task(task_config, sheet_name):
    task_name = task_config["name"]
    pretrain_file = task_config["pretrain_file"]

    print("\n" + "=" * 100)
    print(f"开始独立运行任务：{task_name}")
    print(f"数据Sheet：{sheet_name}")
    print(f"预训练文件：{pretrain_file}")
    print("=" * 100)

    # 每个任务重新固定相同随机种子，确保三个任务分别独立运行
    set_random_seed(BASE_SEED)

    # 1. 当前任务独立预训练
    print(f"\n===== {task_name}：预训练阶段 =====")
    encoder, scaler, imputer, pretrain_features = pretrain_feature_encoder(dataset_path=pretrain_file, epochs=50)

    # 2. 当前任务独立读取对应Sheet
    print(f"\n===== {task_name}：加载开发数据 =====")
    task_data = pd.read_excel(DEVELOPMENT_FILE, sheet_name=sheet_name)
    X, y, train_features = prepare_dataset(task_data, task_name)

    print(f"{task_name}数据维度: {X.shape}")
    print(f"N={(y == 0).sum()}, T={(y == 1).sum()}")

    # 3. 当前任务独立初始化特征提取器
    feature_extractor = UnifiedFeatureExtractor(encoder, scaler, imputer, pretrain_features, train_features)

    # 4. 当前任务独立初始化TRAIN模型
    model = TabPFNClassifier(device=device, model_path=MODEL_PATH)

    # 5. 当前任务独立进行10次70/30划分
    accuracies = []
    auprcs = []
    precisions_list = []
    recalls_list = []

    print(f"\n===== TRAIN模型：{task_name} =====")

    for i in range(N_REPEATS):
        print(f"\n{task_name}迭代 {i + 1}/{N_REPEATS}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=i)
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

        X_train_enc, X_test_enc = feature_extractor.transform(X_train, X_test)

        start_time = time.time()
        model.fit(X_train_enc, y_train)
        print(f"训练时间: {time.time() - start_time:.2f}秒")

        y_pred = model.predict(X_test_enc)
        y_pred_proba = model.predict_proba(X_test_enc)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        auprc = auc(recall, precision)

        accuracies.append(accuracy)
        auprcs.append(auprc)
        precisions_list.append(precision)
        recalls_list.append(recall)

        print(f"{task_name}迭代 {i + 1} 性能: 准确率={accuracy:.3f}, AUPRC={auprc:.3f}")

    # 6. 当前任务计算平均PRC
    mean_recall = np.linspace(0, 1, 100)
    interp_precisions = []

    for precision, recall in zip(precisions_list, recalls_list):
        idx_sort = np.argsort(recall)
        interp_precisions.append(np.interp(mean_recall, recall[idx_sort], precision[idx_sort]))

    mean_precision = np.mean(interp_precisions, axis=0)
    std_precision = np.std(interp_precisions, axis=0)

    result = {
        "name": task_name,
        "sheet_name": sheet_name,
        "pretrain_file": pretrain_file,
        "color": task_config["color"],
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_auprc": np.mean(auprcs),
        "std_auprc": np.std(auprcs),
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "std_precision": std_precision
    }

    print("\n" + "-" * 100)
    print(f"{task_name}独立运行完成")
    print(f"平均准确率: {result['mean_accuracy']:.3f} (±{result['std_accuracy']:.3f})")
    print(f"平均AUPRC: {result['mean_auprc']:.3f} (±{result['std_auprc']:.3f})")
    print("-" * 100)

    return result

# ===================== 10. 主函数 =====================
def main():
    # 1. 检查文件
    required_files = [DEVELOPMENT_FILE, MODEL_PATH] + [task["pretrain_file"] for task in TASK_CONFIGS]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件：{file_path}")

    # 2. 获取开发文件Sheet顺序
    excel_file = pd.ExcelFile(DEVELOPMENT_FILE)
    sheet_names = excel_file.sheet_names

    if len(sheet_names) < 3:
        raise ValueError(f"{DEVELOPMENT_FILE}至少需要3个Sheet，当前只有{len(sheet_names)}个：{sheet_names}")

    selected_sheets = sheet_names[:3]

    print("\n===== 三个二分类任务配置 =====")

    for i, task in enumerate(TASK_CONFIGS):
        print(f"{i + 1}. {task['name']} → Sheet: {selected_sheets[i]} → Pretrain: {task['pretrain_file']}")

    # 3. 三个任务依次完全独立运行
    all_results = []

    for i, task_config in enumerate(TASK_CONFIGS):
        result = run_single_task(task_config, selected_sheets[i])
        all_results.append(result)

    # 4. 打印三个任务最终结果
    print("\n" + "=" * 100)
    print("===== TRAIN三个疾病对开发集AUPRC结果 =====")
    print("=" * 100)

    for result in all_results:
        print(f"{result['name']}:")
        print(f"  Sheet: {result['sheet_name']}")
        print(f"  Pretrain: {result['pretrain_file']}")
        print(f"  平均准确率: {result['mean_accuracy']:.3f} (±{result['std_accuracy']:.3f})")
        print(f"  平均AUPRC: {result['mean_auprc']:.3f} (±{result['std_auprc']:.3f})")
        print("-" * 50)

    # 5. 绘制三个疾病对的平均PRC曲线
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 22, "axes.titlesize": 22, "axes.labelsize": 22, "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 17})
    plt.figure(figsize=(8, 6))

    for result in all_results:
        lower_precision = np.clip(result["mean_precision"] - result["std_precision"], 0, 1)
        upper_precision = np.clip(result["mean_precision"] + result["std_precision"], 0, 1)
        plt.plot(result["mean_recall"], result["mean_precision"], lw=2, label=f"{result['name']} AUPRC: {result['mean_auprc']:.3f} (±{result['std_auprc']:.3f})", color=result["color"])
        plt.fill_between(result["mean_recall"], lower_precision, upper_precision, alpha=0.08, color=result["color"])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test: Clinical Pairwise Classification")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()