# By Shuran Liu, 2023
import random
import numpy as np

from torch.utils.data import Dataset
import h5py
import cv2

class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale
        self.z_size = 8  # modified by xwc 0915
        

    @staticmethod
    def random_crop(lr, hr, size, scale, z_size):
        lr_left = random.randint(0, lr.shape[4] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[3] - size)
        lr_bottom = lr_top + size
        lr_up = random.randint(0, lr.shape[2] - z_size)
        lr_down = lr_up + z_size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[:, :, lr_up:lr_down, lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[:, :, lr_up:lr_down, hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr
    @staticmethod
    def center_crop(lr, hr, patch_size, scale):
        _, _,_, width, height = lr.shape
        left = (width - patch_size) // 2
        top = (height - patch_size) // 2
        right = (width + patch_size) // 2
        bottom = (height + patch_size) // 2
        lr = lr[:,:,:,left:right, top:bottom]
        hr = hr[:,:,:,scale*left:scale*right, scale*top:scale*bottom]
        return lr, hr

    @staticmethod
    def random_rotate(lr, hr, scale):
        angle = random.uniform(-15,15)
        batch_size, time_steps, depth,width, height = lr.shape
        rotation_matrix_lr = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotation_matrix_hr = cv2.getRotationMatrix2D((scale*width / 2, scale*height / 2), angle, 1)
        for i in range(batch_size):
            for j in range(time_steps):
                for z in range(depth):
                    lr[i,j,z] = cv2.warpAffine(lr[i,j,z], rotation_matrix_lr, (width, height))
                    hr[i,j,z] = cv2.warpAffine(hr[i,j,z], rotation_matrix_hr, (scale*width, scale*height))
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, :, :, :, ::-1].copy()
            hr = hr[:, :, :, :, ::-1].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, :, :, ::-1, :].copy()
            hr = hr[:, :, :, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(4, 3)).copy()
            hr = np.rot90(hr, axes=(4, 3)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)]
            lr = np.array(lr)
            gt = f['hr'][str(idx)]
            gt = np.array(gt)
            lr, gt= self.random_crop(lr, gt, int(self.patch_size*1.3), self.scale, self.z_size)
            lr, gt = self.random_rotate(lr,gt,self.scale)
            lr, gt = self.center_crop(lr, gt, self.patch_size, self.scale)
            lr, gt = self.random_vertical_flip(lr, gt)
            lr, gt = self.random_rotate_90(lr, gt)
            return lr.astype(np.float32), gt.astype(np.float32)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])