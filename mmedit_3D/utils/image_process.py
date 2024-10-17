import numpy as np
import math
import cv2
import torch
from cv2 import GaussianBlur, BORDER_REPLICATE
from utils.pytorch_msssim import ms_ssim
device = torch.device('cuda')

# --------------------------------------------
# Calculate the average number of photons from raw data
# --------------------------------------------
def F_cal_average_photon(x, Gaussian_sigma=5.0, ratio=0.2, conversion_factor=0.6026, mean_axis=None):  # default Gaussian_sigma=5.0 which is too large
    # (0) get the average image (wide-field)
    assert x.ndim == 7
    if mean_axis: x = np.mean(x, axis=mean_axis) # [TOPCD]HW
    # (1) subtracting the average background of the camera
    x[x < 0] = 0
    # (2) using a Gaussian LPF (sigma=5) to blur the original image, ksize is set to 9 as following qiao's advice
    x_blur = GaussianBlur(src=x, ksize=(9, 9), sigmaX=Gaussian_sigma, sigmaY=Gaussian_sigma, borderType=BORDER_REPLICATE)
    # (3) performing the percentile-normalization on the filtered image
    #     extracting the feature-only regions of the normalized image with threshold 0.2
    x = x[x_blur > np.max(x_blur) * ratio]
    # (4) calculating the average sCMOS count of the thresholded image
    average_sCMOS_count = np.mean(x)
    # (5) converting the sCMOS count into the photon count by a conversion factor of 0.6026 photons per count, which is measured via Hamamatsu's protocol
    average_photon_count = average_sCMOS_count * conversion_factor
    return average_photon_count

# --------------------------------------------
# running average time-axis filtering
# --------------------------------------------
def running_average(x, t=3):
    if x.shape[0] <= 2: return x
    if t == 1: return x
    if t % 2 == 0: t += 1
    b = x.shape[0]
    mean_list = []
    # Report x.mean()
    for idx in range(b):
        mean_list.append(np.mean(x[idx, ...]))
    y = np.zeros_like(x)
    # Do running average
    for idx in range(0,t//2):
        y[idx, ...] = np.mean(x[:t, ...], axis=0)
    for idx in range(t//2,b-t//2):
        y[idx, ...] = np.mean(x[idx-t//2:idx+t//2+1, ...], axis=0)
    for idx in range(b-t//2, b):
        y[idx, ...] = np.mean(x[-t:, ...], axis=0)
    # Intensity Norm
    for idx in range(b):
        y[idx, ...] = y[idx, ...] / np.mean(y[idx, ...]) * mean_list[idx]
    return y

# --------------------------------------------
# evaluation index MS-SSIM
# --------------------------------------------
def calculate_ms_ssim_peak(img1, img2, border=30):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    img1_stack = img1.reshape(-1, h, w)
    img2_stack = img2.reshape(-1, h, w)

    msssim_sum = 0
    msssim_num = 0
    for idx in range(img1_stack.shape[0]):
        img1 = img1_stack[idx, border:h - border, border:w - border]
        img2 = img2_stack[idx, border:h - border, border:w - border]
        img1 = torch.from_numpy(img1).to(device)
        img2 = torch.from_numpy(img2).to(device)
        ms_ssim_val = ms_ssim(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0), data_range=torch.max(img1), size_average=True).cpu().numpy().astype(np.float32)
        msssim_sum += ms_ssim_val
        msssim_num += 1

    return msssim_sum/msssim_num

# --------------------------------------------
# evaluation index nrmse
# --------------------------------------------
def calculate_nrmse(img1, img2, border=30):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    img1 = img1[..., border:h-border, border:w-border]
    img2 = img2[..., border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if np.max(img1) - np.min(img1) == 0: return float('inf')
    return math.sqrt(mse) / (np.max(img1) - np.min(img1))

# --------------------------------------------
# evaluation index psnr
# --------------------------------------------
def calculate_psnr_peak(img1, img2, border=30):
    # img1 is the ground-truth image
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    img1 = img1[..., border:h-border, border:w-border]
    img2 = img2[..., border:h-border, border:w-border]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    peak = np.max(img1)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(peak / math.sqrt(mse))


# --------------------------------------------
# evaluation index ssim
# --------------------------------------------
def calculate_ssim_peak(img1, img2, border=30):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    img1_stack = img1.reshape(-1, h, w)
    img2_stack = img2.reshape(-1, h, w)

    ssim_num = 0
    ssim_sum = 0

    for idx in range(img1_stack.shape[0]):
        img1 = img1_stack[idx, border:h-border, border:w-border]
        img2 = img2_stack[idx, border:h-border, border:w-border]
        peak = np.max(img1)
        C1 = (0.01 * peak) ** 2
        C2 = (0.03 * peak) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) \
                   / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_sum += ssim_map.mean()
        ssim_num += 1

    return ssim_sum / ssim_num

# --------------------------------------------
# alignment input image to have same background and intensity
# --------------------------------------------
def Align2Reference(im_in,im_GT):
    numb=len(im_GT.flatten())
    k=(np.sum(im_in*im_GT)-np.mean(im_in)*np.mean(im_GT)*numb) / (np.sum(im_in*im_in)- np.power(np.mean(im_in),2)*numb)
    b= np.mean(im_GT) - k*np.mean(im_in)
    im_out=k*im_in + b
    print('k='+str(k)+' b='+str(b))
    return im_out

def cal_maxInt(im,percent=0.005,keepdim=False):
    shaperaw = im.shape
    if len(im.shape)<3:
        im = torch.unsqueeze(im,dim=0)
    shapein=im.shape
    framenumb=torch.prod(torch.tensor(shapein[0:-2]))
    pixelnumb=torch.prod(torch.tensor(shapein[-2:]))
    im = im.reshape((framenumb,pixelnumb))
    im2=torch.sort(im,dim=1,descending=True).values
    pixelnumb_threshold=torch.ceil(pixelnumb*percent).to(torch.int64)
    im3=torch.mean(im2[:,0:pixelnumb_threshold],dim=1)
    if keepdim:
        im3=im3.reshape((*(shaperaw[0:-2]), 1, 1))
    else:
        im3=im3.reshape(shaperaw[0:-2])
    return im3
