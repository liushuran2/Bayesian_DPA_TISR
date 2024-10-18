import time
from functools import wraps
from contextlib import contextmanager
import datetime
from functools import reduce
import numpy as np
import random
import torch
import os
import logging

# --------------------------------------------
# time this
# --------------------------------------------
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper

# --------------------------------------------
# timeblock
# --------------------------------------------
@contextmanager
def timeblock(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start))

# --------------------------------------------
# imshow
# --------------------------------------------
# def sp_imshow(x, cmap='gray'):
#     if isinstance(x, torch.Tensor): x = x.cpu().numpy()
#     x = x.squeeze()
#     if x.dtype in [np.complex64, np.complex128]: x = np.real(x)
#     plt.imshow(x, cmap=cmap)
#     plt.show()
#
# def fq_imshow(x, cmap='gray'):
#     if isinstance(x, torch.Tensor): x = x.cpu().numpy()
#     x = x.squeeze()
#     x = np.log(np.abs(x))
#     plt.imshow(x, cmap=cmap)
#     plt.show()
#
# def sp_fft2d_imshow(x):
#     if not isinstance(x, torch.Tensor): raise NotImplementedError("input in sp_fft2d_imshow must be tensor")
#     x = torch.fft.ifftshift(x, [-1,-2])
#     x = torch.fft.fft2(x)
#     x = torch.fft.fftshift(x, [-1,-2])
#     fq_imshow(x)
#
# def fq_ifft2d_imshow(x):
#     if not isinstance(x, torch.Tensor): raise NotImplementedError("input in sp_fft2d_imshow must be tensor")
#     x = torch.fft.ifftshift(x, [-1,-2])
#     x = torch.fft.fft2(x)
#     x = torch.fft.fftshift(x, [-1,-2])
#     sp_imshow(x)


# --------------------------------------------
# output in txt
# --------------------------------------------
class WriteOutputTxt:
    def __init__(self, filename, encoding=None, if_print=True, if_use=True):
        self.if_use = if_use
        if if_use:
            # check
            assert filename[-4:] == '.txt'
            self.filename = filename
            self.encoding = encoding
            self.if_print = if_print
            with open(filename, mode="a", encoding=encoding) as f:
                f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

    def info(self, content):
        if self.if_use:
            if self.if_print: print(content)
            with open(self.filename, mode="a", encoding=self.encoding) as f:
                f.write(content)

# --------------------------------------------
# dir tool
# --------------------------------------------
def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_with_time(path):
    path += '_' + get_timestamp()
    os.makedirs(path)
    return path

# --------------------------------------------
# tuple/list product
# --------------------------------------------
def product_of_tuple_elements(tup):
    return reduce(lambda x,y:x*y, tup)


# --------------------------------------------
# seed
# --------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def permutate(number):
    from itertools import permutations
    nums=list(np.arange(number))
    result = []
    #result.append(nums)
    for i in permutations(nums, number):
        if not sum((nums - np.array(i)) == 0):
            result.append(i)

    return result

def topk_np(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort
