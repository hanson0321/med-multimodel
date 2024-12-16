import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# 讀取 Metabolomics 數據
metabolomics_file_path = 'Metabolomics_n102.csv'
metabolomics_data = pd.read_csv(metabolomics_file_path)

# 1️⃣ **處理缺失值**

# 計算每列的缺失率
missing_rate = metabolomics_data.isnull().mean()

# 刪除缺失率超過50%的列
columns_to_drop = missing_rate[missing_rate > 0.5].index
metabolomics_data = metabolomics_data.drop(columns=columns_to_drop)
print(f"刪除的缺失率 > 50% 的列: {list(columns_to_drop)}")

# 對數值型數據的缺失值進行填充（用中位數填充）
numeric_cols = metabolomics_data.select_dtypes(include=['float64', 'int64']).columns
metabolomics_data[numeric_cols] = metabolomics_data[numeric_cols].fillna(metabolomics_data[numeric_cols].median())

# 2️⃣ **標準化數據**

# 標準化數值變數
scaler = StandardScaler()
metabolomics_data[numeric_cols] = scaler.fit_transform(metabolomics_data[numeric_cols])

# 3️⃣ **特徵選擇**

# 移除方差過低的特徵（例如變異性小於0.01的特徵）
selector = VarianceThreshold(threshold=0.01)
numeric_data = metabolomics_data[numeric_cols]
reduced_numeric_data = selector.fit_transform(numeric_data)

# 更新 DataFrame，僅保留有較高變異的數據
kept_columns = numeric_cols[selector.get_support()]
metabolomics_data = metabolomics_data[kept_columns]
print(f"保留的特徵數量: {len(kept_columns)} / {len(numeric_cols)}")

# 4️⃣ **刪除重複的行**
metabolomics_data = metabolomics_data.drop_duplicates()

# 5️⃣ **數據檢查**

# 顯示預處理後的數據的前5行
print("\n預處理後的 Metabolomics 數據的前5行：")
print(metabolomics_data.head())

# 檢查預處理後的數據信息
print("\n預處理後的 Metabolomics 數據的信息：")
metabolomics_info_after = metabolomics_data.info()

# 6️⃣ **保存預處理後的數據**

# 將清理後的數據保存為CSV文件
processed_file_path = 'processed_metabolomics_data.csv'
metabolomics_data.to_csv(processed_file_path, index=False)
print(f"\n已將預處理後的 Metabolomics 數據保存至 {processed_file_path}")
