# By Shuran Liu, 2023
import h5py
import numpy as np
from skimage import io


output = 'XXX.h5'


h5_file = h5py.File(output, 'w')

lr_group = h5_file.create_group('lr')
hr_group = h5_file.create_group('hr')

patch_idx = 0
n_frame = 7
path = ''
total_num = 50
root = path + '/train'
patchsize = 192
factor = 2
num_in_seq = 10
for i in range(1,total_num + 1):
    for seq_num in range(num_in_seq):
        hr=np.zeros((n_frame,1,patchsize*factor,patchsize*factor)).astype(np.float64)
        lr=np.zeros((n_frame,1,patchsize,patchsize)).astype(np.float64)
        for k in range(n_frame):
            hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(k+seq_num+1).rjust(4, '0') +'.tif')
            lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_step4'+'/' + str(k+seq_num+1).rjust(3, '0') +'.tif')
            hr_temp = np.array(hr_temp, dtype=np.float32)
            lr_temp = np.array(lr_temp, dtype=np.float32)
            hr_temp = hr_temp[:, :, np.newaxis]
            lr_temp = lr_temp[:, :, np.newaxis]
            hr[k,:,:,:]=np.asarray(hr_temp).astype(np.float32).transpose(2,0,1)
            lr[k,:,:,:]=np.asarray(lr_temp).astype(np.float32).transpose(2,0,1)

        lr_group.create_dataset(str(patch_idx), data=lr)
        hr_group.create_dataset(str(patch_idx), data=hr)
        
        patch_idx += 1
        print(patch_idx)

h5_file.close()