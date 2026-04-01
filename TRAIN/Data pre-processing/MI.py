import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 必须保留
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

def fill_na_with_multiple_imputation(input_file, output_file, missing_threshold=0.5):
    """
    优化后的多重插补：处理临床特征缺失值
    
    参数:
    - input_file: 输入文件名
    - output_file: 输出文件名
    - missing_threshold: 缺失率阈值，默认超过50%缺失的列将被剔除，防止引入过多噪音
    """
    try:
        # 1. 读取数据：统一缺失值标识
        print(f"正在读取文件: {input_file}")
        df = pd.read_excel(input_file, header=0, index_col=0, na_values=['#N/A', '-', 'nan', 'NULL'])
        
        # 2. 缺失率预检 (科学性：删除缺失过多的特征)
        missing_rates = df.isnull().mean()
        cols_to_drop = missing_rates[missing_rates > missing_threshold].index.tolist()
        if cols_to_drop:
            print(f"警告: 以下列缺失率超过 {missing_threshold*100}%，已剔除以保证逻辑严谨: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # 3. 分类变量与数值变量识别
        # 假设第一列（即 df.columns[0]）是性别或分类变量
        cat_column = df.columns[0]
        num_columns = df.columns[1:]
        
        df_processing = df.copy()

        # 4. 严谨处理分类变量 (性别)
        # 临床数据中性别是关键协变量，需编码后参与插补计算，最后四舍五入还原
        le = LabelEncoder()
        # 仅对非空值编码，避免干扰插补器
        mask = df_processing[cat_column].notnull()
        df_processing.loc[mask, cat_column] = le.fit_transform(df_processing[cat_column][mask])
        
        # 5. 强制数值转换
        for col in num_columns:
            df_processing[col] = pd.to_numeric(df_processing[col], errors='coerce')

        # 6. 配置 IterativeImputer (核心逻辑优化)
        # 优化点：
        # - min_value=0: 确保临床指标（如年龄、指标浓度）不会出现不符合物理意义的负数
        # - max_iter=20: 增加迭代次数确保收敛
        imputer = IterativeImputer(
            max_iter=20,
            random_state=42,
            min_value=0,          # 限制最小值，防止出现负数浓度
            initial_strategy='median', # 初始值使用中位数比均值更稳健
            skip_complete=True
        )

        print("正在进行多重插补计算（MICE逻辑）...")
        imputed_data = imputer.fit_transform(df_processing)

        # 7. 还原数据框
        df_final = pd.DataFrame(imputed_data, columns=df_processing.columns, index=df_processing.index)

        # 8. 还原分类变量逻辑：使用 round() 确保 0.6 变 1，0.4 变 0，而非直接截断
        df_final[cat_column] = df_final[cat_column].round().astype(int)
        df_final[cat_column] = le.inverse_transform(df_final[cat_column])

        # 9. 结果验证与保存
        print(f"插补完成。原始缺失值总数: {df.isnull().sum().sum()}")
        print(f"处理后剩余缺失值数量: {df_final.isnull().sum().sum()}")
        
        df_final.to_excel(output_file)
        print(f"结果已保存至: {output_file}")
        
        return df_final

    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None

if __name__ == "__main__":
    fill_na_with_multiple_imputation("2342样本73特征.xlsx", "2342样本73特征_已填补.xlsx")