# By Shuran Liu, 2023
import random
import numpy as np

from torch.utils.data import Dataset
import h5py
import cv2

# from prepare_data import z_length


class TestDataset(Dataset):
    def __init__(self, h5_file, scale):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = scale

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)]
            lr = np.array(lr)
            hr = f['hr'][str(idx)]
            gt = np.array(hr)
            return lr.astype(np.float32), gt.astype(np.float32)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])