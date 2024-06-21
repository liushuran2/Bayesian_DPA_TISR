# By Shuran Liu, 2023
import argparse
import yaml
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)
import numpy as np
from torch.utils.data import DataLoader
from Testdataset import TestDataset
import torch
from mmedit.models.backbones.sr_backbones import DPATISR
from diagram import reliability_diagram, interval_confidence
import loss
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

from scipy.optimize import leastsq

def linear_error(p,gt,pre):
    return p[0]*pre+p[1]-gt
    
def linear_trans(gt,pre):
    p0 = [1,0]
    result = leastsq(linear_error,p0,args=(gt.ravel(),pre.ravel()))
    a,b = result[0]
    return a*pre+b

def calc_psnr(sr, hr, scale=3, rgb_range=255, dataset=None):
    sr = linear_trans(hr, sr)
    diff = (sr - hr)
    mse = np.mean((diff) ** 2)
    return -10 * math.log10(mse)

def calc_ssim(sr,hr):
    sr = linear_trans(hr, sr)
    return ssim(sr,hr)

def mkdir(path):
 
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)
mkdir(config['save_path'])
mkdir(config['save_path'] + '/Confidence')
mkdir(config['save_path'] + '/Data')
mkdir(config['save_path'] + '/Model')
mkdir(config['save_path'] + '/SR')

epsilon = 0.04
model = torch.nn.DataParallel(DPATISR(mid_channels=config['mid_channels'],
                 extraction_nblocks=config['extraction_nblocks'],
                 propagation_nblocks=config['propagation_nblocks'],
                 reconstruction_nblocks=config['reconstruction_nblocks'],
                 factor=config['factor'],
                 bayesian=config['bayesian'])).cuda()

if config['bayesian']:
    num_dropout_ensembles = 6
else:
    num_dropout_ensembles = 1
checkpt=torch.load(config['inference_checkpt'])
model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

test_dataset = TestDataset(config['test_dataset_path'], scale=2)
dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                    num_workers=config['num_workers'],
                        pin_memory=True)
count = 0
patch_size=512
factor = config['factor']
with torch.no_grad():
    model.eval()
    if config['bayesian']:
        enable_dropout(model)
    with tqdm(dataloader, desc="Training MANA") as tepoch:
        psnr_list=[]
        ssim_list=[]
        for inp, gt in tepoch:
            mean = np.ndarray((num_dropout_ensembles, factor*patch_size, factor*patch_size)) #这里要改
            data_uncertainty = np.ndarray((num_dropout_ensembles, factor*patch_size, factor*patch_size))
            count += 1
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            gt = gt[:,3,:,:,:]
            for i in range(num_dropout_ensembles):
                oup = model(inp)
                oup = oup[:,3,:,:,:]
                SR_y = oup[0, 0:1, :, :].permute(1, 2, 0).data.cpu().numpy()
                SR_y = SR_y.astype(np.float32)
                SR_y = SR_y.squeeze(2)
                if config['bayesian']:
                    std_y = oup[0, 1:2, :, :].permute(1, 2, 0).data.cpu().numpy()
                    std_y = std_y.astype(np.float32)
                    std_y = std_y.squeeze(2)
                mean[i, :, :] = SR_y
                if config['bayesian']:
                    data_uncertainty[i, :, :] = std_y

            if config['bayesian']:
                bin_confidence, bin_correct ,bin_total= reliability_diagram(gt.data.cpu().numpy(), mean, data_uncertainty, epsilon)
                bin_correct = np.array(bin_correct)
                bin_confidence = np.array(bin_confidence)
                bin_total = np.array(bin_total)
                if count == 1:
                    bin_confidences = bin_confidence
                    bin_corrects = bin_correct
                    bin_totals = bin_total
                else:
                    bin_confidences += bin_confidence
                    bin_corrects += bin_correct
                    bin_totals += bin_total
            SR_result = np.mean(mean, axis=0)
            if config['bayesian']:
                data_uncertainty_result = np.mean(data_uncertainty, axis=0)
                model_uncertainty_result = np.std(mean, axis=0)

            gt = gt.squeeze(0)
            gt = gt.squeeze(0)
            gt = gt.cpu().detach().numpy()

            psnr_list.append(calc_psnr(SR_result, gt))
            ssim_list.append(calc_ssim(SR_result, gt))

            SR_result = Image.fromarray(SR_result)
            SR_result.save(config['save_path'] + '/SR/super_res{}.tif'.format(str(count)))

            if config['bayesian']:
                data_uncertainty_result = Image.fromarray(data_uncertainty_result)
                data_uncertainty_result.save(config['save_path'] + '/Data/datauncer{}.tif'.format(str(count)))

                model_uncertainty_result = Image.fromarray(model_uncertainty_result)
                model_uncertainty_result.save(config['save_path'] + '/Model/modeluncer{}.tif'.format(str(count)))

                confidence = interval_confidence(mean, data_uncertainty, config['epsilon'], num_dropout_ensembles)
                confidence = Image.fromarray(confidence)
                confidence.save(config['save_path'] + '/Confidence/confidence{}.tif'.format(str(count)))

    if config['bayesian']:
        non_zero = bin_totals.nonzero()
        non_zero = np.where(bin_totals > 100)
        bin_confidences = bin_confidences[non_zero] / bin_totals[non_zero]
        bin_corrects = bin_corrects[non_zero] / bin_totals[non_zero]
        bin_ratio = bin_totals[non_zero] / np.sum(bin_totals[non_zero])
        ECE = np.dot(np.abs(bin_confidences - bin_corrects), bin_ratio)
        print(ECE)

    print(np.mean(psnr_list))
    print(np.mean(ssim_list))

if config['bayesian']:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal')
    ax.plot(bin_confidences, bin_corrects, marker='o', label='Model')
    ax.set_xlabel('Average Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    plt.show()
    plt.savefig('result/Reliable-diagram.png')
