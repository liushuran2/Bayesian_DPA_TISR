import time
from functools import wraps
from contextlib import contextmanager
import matplotlib.pyplot as plt
import torch
import numpy as np
"""
basic tools
"""

# --------------------------------------------
# dict to string for logger by recursion
# --------------------------------------------
def dict2str(opt, indent_l=1):
    msg = ''
    for key, vaule in opt.items():
        if isinstance(vaule, dict):
            msg += ' ' * (indent_l * 2) + key + ':{\n'
            msg += dict2str(vaule, indent_l + 1)
            msg += ' ' * (indent_l * 2) + '}\n'
        else:
            msg += ' ' * (indent_l * 2) + key + ': ' + str(vaule) + '\n'
    return msg

# --------------------------------------------
# print dict structurally
# --------------------------------------------
def print_dict(opt):
    print(dict2str(opt))

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
def sp_imshow(x):
    if isinstance(x, torch.Tensor): x = x.cpu().numpy()
    x = x.squeeze()
    x = np.abs(x)
    plt.imshow(x)
    plt.show()

def fq_imshow(x):
    if isinstance(x, torch.Tensor): x = x.cpu().numpy()
    x = x.squeeze()
    x = np.log(np.abs(x))
    plt.imshow(x)
    plt.show()

def sp_fft2d_imshow(x):
    if not isinstance(x, torch.Tensor): raise NotImplementedError("input in sp_fft2d_imshow must be tensor")
    x = torch.fft.ifftshift(x, [-1,-2])
    x = torch.fft.fft2(x)
    x = torch.fft.fftshift(x, [-1,-2])
    fq_imshow(x)

def fq_ifft2d_imshow(x):
    if not isinstance(x, torch.Tensor): raise NotImplementedError("input in sp_fft2d_imshow must be tensor")
    x = torch.fft.ifftshift(x, [-1,-2])
    x = torch.fft.fft2(x)
    x = torch.fft.fftshift(x, [-1,-2])
    sp_imshow(x)



if __name__ == '__main__':
    print_dict({'name': 1, 'idx': {'hello': 3, 'world': {'hello': 3, 'world': 4}}, 'haha': 2})