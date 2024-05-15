# By Shuran Liu, 2023
import random
import numpy as np

from torch.utils.data import Dataset
import h5py
import cv2

class TestDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = scale

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)]
            lr = np.array(lr)
            gt = f['hr'][str(idx)]
            gt = np.squeeze(gt[:, :, :, :], axis=1)
            for i in range(gt.shape[0]):
                gt_temp = gt[i]
                gt_temp = cv2.GaussianBlur(gt_temp, (3,3), 0.8)
                gt[i] = gt_temp
            gt = np.expand_dims(gt, 1)
            return lr.astype(np.float32), gt.astype(np.float32)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])