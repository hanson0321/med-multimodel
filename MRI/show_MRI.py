import os
import nibabel as nib
import numpy as np

# 檔案路徑 (您需要更新這個路徑到您的本地目錄)
folder_path = 'MRI_demo_data2'  # 替換成您 .nii 檔案的資料夾路徑
output_folder = 'MRI picture'  # 替換成您想要儲存 .npy 檔案的資料夾路徑

# 如果輸出資料夾不存在，則建立它
os.makedirs(output_folder, exist_ok=True)

# 取得所有 .nii 和 .nii.gz 檔案
nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]

# 讀取和儲存每個 NIfTI 檔案的影像數據
for i, file_path in enumerate(nii_files):
    try:
        print(f'正在讀取檔案: {file_path}')
        # 讀取 NIfTI 檔案
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # 產生輸出檔案名稱
        base_name = os.path.basename(file_path).replace('.nii.gz', '').replace('.nii', '')
        output_file_path = os.path.join(output_folder, f'{base_name}_image.npy')
        
        # 儲存影像數據為 .npy 檔案
        np.save(output_file_path, data)
        print(f'儲存成功: {output_file_path}')
    except Exception as e:
        print(f'讀取檔案失敗: {file_path}, 錯誤: {e}')
