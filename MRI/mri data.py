import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

# 定義 MRI 數據處理類
class MRIDataset(Dataset):
    def __init__(self, mri_folder, labels=None, transform=None):
        """
        Args:
            mri_folder (str): 存放 MRI .npy 文件的資料夾路徑。
            labels (list or None): 如果有標籤，則傳入對應的標籤列表。
            transform (callable, optional): 可選的數據增強或轉換。
        """
        self.mri_file_paths = [os.path.join(mri_folder, file) for file in os.listdir(mri_folder) if file.endswith('.npy')]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.mri_file_paths)

    def __getitem__(self, idx):
        # 加載 MRI 數據
        mri_data = np.load(self.mri_file_paths[idx])
        
        # 確保數據形狀符合模型需求 (例如 [C, H, W])
        if len(mri_data.shape) == 3:
            mri_data = np.expand_dims(mri_data, axis=0)  # 增加通道維度

        # 應用數據轉換
        if self.transform:
            mri_data = self.transform(mri_data)

        # 將數據轉換為 Tensor
        mri_data = torch.tensor(mri_data, dtype=torch.float32)

        # 返回數據和標籤（如果有）
        if self.labels is not None:
            label = self.labels[idx]
            return mri_data, label
        return mri_data

# 定義數據處理函數
def load_mri_data_from_folder(mri_folder, labels=None, batch_size=16, shuffle=True):
    """
    加載 MRI 數據集並創建 DataLoader。

    Args:
        mri_folder (str): 存放 MRI .npy 文件的資料夾路徑。
        labels (list or None): 標籤數據。
        batch_size (int): 每批數據量。
        shuffle (bool): 是否打亂數據順序。

    Returns:
        DataLoader: PyTorch 的數據加載器。
    """
    dataset = MRIDataset(mri_folder, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 示例：加載 MRI 數據
mri_folder = "MRI picture"  # 替換為實際資料夾路徑
labels = [0] * 100  # 替換為對應的標籤，這裡假設100個樣本全為標籤0

dataloader = load_mri_data_from_folder(mri_folder, labels, batch_size=8)

# 遍歷數據加載器檢查數據
for batch in dataloader:
    mri_data, mri_labels = batch
    print(f"MRI Data Shape: {mri_data.shape}, Labels: {mri_labels}")
    break
