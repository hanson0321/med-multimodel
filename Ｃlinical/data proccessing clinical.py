import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 讀取 Clinical 數據
clinical_file_path = 'Clinical_n102.xlsx - clinical.csv'
clinical_data = pd.read_csv(clinical_file_path)

# 1️⃣ **處理缺失值**

# 計算每列的缺失率
missing_rate = clinical_data.isnull().mean()

# 刪除缺失率超過50%的列
columns_to_drop = missing_rate[missing_rate > 0.5].index
clinical_data = clinical_data.drop(columns=columns_to_drop)
print(f"刪除的缺失率 > 50% 的列: {list(columns_to_drop)}")

# 對數值型數據的缺失值進行填充（用中位數填充）
numeric_cols = clinical_data.select_dtypes(include=['float64', 'int64']).columns
clinical_data[numeric_cols] = clinical_data[numeric_cols].fillna(clinical_data[numeric_cols].median())

# 對類別型數據的缺失值進行填充（用眾數填充）
categorical_cols = clinical_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    most_frequent = clinical_data[col].mode()[0]
    clinical_data[col] = clinical_data[col].fillna(most_frequent)

# 2️⃣ **類別變數編碼**

# 對所有的類別變數進行 Label Encoding
for col in categorical_cols:
    label_encoder = LabelEncoder()
    clinical_data[col] = label_encoder.fit_transform(clinical_data[col])

# 3️⃣ **數據標準化**

# 標準化數值變數
scaler = StandardScaler()
clinical_data[numeric_cols] = scaler.fit_transform(clinical_data[numeric_cols])

# 4️⃣ **刪除重複的行**
clinical_data = clinical_data.drop_duplicates()

# 5️⃣ **數據檢查**

# 顯示預處理後的數據的前5行
print("\n預處理後的 Clinical 數據的前5行：")
print(clinical_data.head())

# 檢查預處理後的數據信息
print("\n預處理後的 Clinical 數據的信息：")
clinical_info_after = clinical_data.info()

# 6️⃣ **保存預處理後的數據**

# 將清理後的數據保存為CSV文件
processed_file_path = 'processed_clinical_data.csv'
clinical_data.to_csv(processed_file_path, index=False)
print(f"\n已將預處理後的 Clinical 數據保存至 {processed_file_path}")
