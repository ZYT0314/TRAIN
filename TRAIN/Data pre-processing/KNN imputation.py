import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def fill_na_with_knn(input_file, output_file, n_neighbors=5, missing_threshold=0.5):
    """
    基于 KNN 的临床数据严谨插补
    
    参数:
    - n_neighbors: 参考的邻居数量，临床研究通常取 3-10 之间
    - missing_threshold: 缺失率超过此比例的特征将被剔除
    """
    try:
        # 1. 读取数据 (第一列为样本名，第二列为性别/分组)
        print(f"正在读取文件: {input_file}")
        df = pd.read_excel(input_file, header=0, index_col=0, na_values=['#N/A', '-', 'nan'])
        
        # 2. 预处理：剔除缺失过多的特征
        missing_rates = df.isnull().mean()
        cols_to_drop = missing_rates[missing_rates > missing_threshold].index.tolist()
        if cols_to_drop:
            print(f"剔除高缺失特征: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # 3. 分离分类变量与数值变量
        cat_col = df.columns[0]  # 性别列
        num_cols = df.columns[1:] # 14个临床特征
        
        df_processing = df.copy()

        # 4. 分类变量编码 (Label Encoding)
        le = LabelEncoder()
        mask = df_processing[cat_col].notnull()
        df_processing.loc[mask, cat_col] = le.fit_transform(df_processing[cat_col][mask])

        # 5. 【严谨性核心】标准化 (Standardization)
        # KNN 基于欧式距离，必须先标准化，否则单位大的指标会主导距离计算
        scaler = StandardScaler()
        # 注意：StandardScaler 可以处理 NaN (计算均值时自动忽略)
        df_scaled_values = scaler.fit_transform(df_processing)
        df_scaled = pd.DataFrame(df_scaled_values, columns=df_processing.columns, index=df_processing.index)

        # 6. 执行 KNN 插补
        print(f"正在执行 KNN 插补 (K={n_neighbors})...")
        # weights='distance' 表示距离越近的邻居权重越高，更符合临床直觉
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_array = imputer.fit_transform(df_scaled)

        # 7. 【严谨性核心】逆标准化 (Inverse Transform)
        # 将数据还原回原始的临床单位（如年龄还原回50岁，而非0.5）
        final_values = scaler.inverse_transform(imputed_array)
        df_final = pd.DataFrame(final_values, columns=df_processing.columns, index=df_processing.index)

        # 8. 后处理：分类变量还原与物理限制
        # 性别还原：四舍五入并反转编码
        df_final[cat_col] = df_final[cat_col].round().astype(int)
        df_final[cat_col] = le.inverse_transform(df_final[cat_col])
        
        # 物理限制：临床指标不应为负数
        df_final[num_cols] = df_final[num_cols].clip(lower=0)

        # 9. 保存
        df_final.to_excel(output_file)
        print(f"KNN 插补完成！结果已保存至: {output_file}")
        
        return df_final

    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None

if __name__ == "__main__":
    fill_na_with_knn("LC应答不应答.xlsx", "LC应答不应答_KNN填补.xlsx")