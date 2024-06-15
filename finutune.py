import argparse
import yaml
import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)

scale = config['factor']
import numpy as np
from torch.utils.data import DataLoader
from Traindataset import TrainDataset
from Testdataset import TestDataset
import torch
import loss
from mmedit.models.backbones.sr_backbones import DPATISR
from diagram import reliability_diagram, diag_torch
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def laplace_cdf(x, mu_list, b_list):
    cdf = 0.5 * torch.sign(x - mu_list) * (1 - torch.exp(-torch.abs(x - mu_list) / b_list))
    return cdf
def interval_confidence(mu_list, b_list, epsilon):
    left = laplace_cdf(mu_list - epsilon, mu_list, b_list)
    right = laplace_cdf(mu_list + epsilon, mu_list, b_list)
    return right - left

def fitxyparabola(x1, y1, x2, y2, x3, y3):
    if x1 == x2 or x2 == x3 or x3 == x1:
        print(f"Fit fails; two points are equal: x1={x1}, x2={x2}, x3={x3}")
        peak = 0
    else:
        xbar1 = 0.5 * (x1 + x2) #/* middle of x1 and x2 /
        xbar2 = 0.5 * (x2 + x3) # / middle of x2 and x3 /
        slope1 = (y2-y1)/(x2-x1) #/ the slope at (x=xbar1). /
        slope2 = (y3-y2)/(x3-x2) #/ the slope at (x=xbar2). /
        curve = (slope2-slope1) / (xbar2-xbar1) #/ The change in slope per unit of x. /
    if curve == 0:
        print(f"Fit fails; no curvature: r1=({x1},{y1}), r2=({x2},{y2}), r3=({x3},{y3}) slope1={slope1}, slope2={slope2}, curvature={curve}")
        peak = 0
    else:
        peak = xbar2 - slope2/curve #/ the x value where slope = 0 */
    return peak
def test(model, test_dataset, patch_size=64, epsilon=0.1, num_dropout_ensembles=1):
    dataloader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
    count = 0
    with torch.no_grad():
        model.eval()
        enable_dropout(model)
        with tqdm(dataloader, desc="Training MANA") as tepoch:
            for inp, gt in tepoch:
                mean = np.ndarray((num_dropout_ensembles, scale*patch_size, scale*patch_size)) #这里要改
                data_uncertainty = np.ndarray((num_dropout_ensembles, scale*patch_size, scale*patch_size))
                count += 1
                inp = inp.float().cuda()
                gt = gt.float().cuda()
                for i in range(num_dropout_ensembles):
                    oup = model(inp)
                    oup = oup[:,3,::]
                    gt = gt[:,3,::]
                    SR_y = oup[0, 0:1, :, :].permute(1, 2, 0).data.cpu().numpy()
                    SR_y = SR_y.astype(np.float32)
                    SR_y = SR_y.squeeze(2)
                    std_y = oup[0, 1:2, :, :].permute(1, 2, 0).data.cpu().numpy()
                    std_y = std_y.astype(np.float32)
                    std_y = std_y.squeeze(2)
                    mean[i, :, :] = SR_y
                    data_uncertainty[i, :, :] = std_y
                    
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
                
    non_zero = bin_totals.nonzero()
    non_zero = np.where(bin_totals > 100)

    bin_confidences = bin_confidences[non_zero] / bin_totals[non_zero]
    bin_corrects = bin_corrects[non_zero] / bin_totals[non_zero]
    bin_ratio = bin_totals[non_zero] / np.sum(bin_totals[non_zero])

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal')
    ax.plot(bin_confidences, bin_corrects, marker='o', label='Model')
    ax.set_xlabel('Average Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    plt.show()
    plt.savefig('result/Reliable-diagram.png')
    ECE = np.dot(np.abs(bin_confidences - bin_corrects), bin_ratio)
    print(ECE)
    return ECE, np.sum(bin_confidences - bin_corrects)

def finetune(model, loss_fn, train_dataset, test_dataset, epsilon=0.04, alpha=0.1, lamda=0.5, n_bins = 50):
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=config['batchsize'],
                            shuffle=True,
                            num_workers=config['num_workers'],
                            pin_memory=True)
    count = 0
    for epoch in range(0, 30):
        model.train()
        with tqdm(dataloader, desc="Finetuning MANA") as tepoch:
            for inp, gt in tepoch:
                tepoch.set_description(f"Finetuning MANA--Epoch {epoch} Lamda {lamda}")
                if count==0:
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-6, betas=(0.5, 0.999))
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0, last_epoch=-1)

                inp = inp.float().cuda()
                gt = gt.float().cuda()
                
                optimizer.zero_grad()
                oup = model(inp)
                oup = oup[:,3,::]
                gt = gt[:,3,::]

                confidence = interval_confidence(oup[:,0:1,:,:], oup[:,1:2,:,:], epsilon=epsilon)
                acc = torch.logical_and(oup[:,0:1,:,:] > gt - epsilon, oup[:,0:1,:,:] < gt + epsilon)
                acc = acc.type(torch.FloatTensor).float().cuda()
                Av_confidence = torch.mean(confidence)
                
                confidence = confidence.reshape(-1)
                accuracy = acc.reshape(-1)
                ECE = diag_torch(n_bins, confidence, accuracy)
                
                loss_L1 = loss_fn(gt, oup)
                loss_L1 = loss_L1.mean()
                Av_confidence = Av_confidence.mean()
                ECE = ECE.mean()
                loss = loss_L1 * alpha + (1-alpha) * (ECE + Av_confidence * lamda)
                loss.backward()
                optimizer.step()
                count += 1
        scheduler.step()
    ECE, absECE = test(model, test_dataset, 512, epsilon)
    torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/{}'.format(str(int(lamda*100))) + config['checkpoint_name'])
    return ECE, absECE
epsilon = config['epsilon']
os.makedirs(config['checkpoint_folder'], exist_ok=True)
model = torch.nn.DataParallel(DPATISR(mid_channels=config['mid_channels'],
                 extraction_nblocks=config['extraction_nblocks'],
                 propagation_nblocks=config['propagation_nblocks'],
                 reconstruction_nblocks=config['reconstruction_nblocks'],
                 factor=config['factor'])).cuda()

checkpt=torch.load(config['hot_start_checkpt'])
model.module.load_state_dict(checkpt)

train_dataset = TrainDataset(config['train_dataset_path'], patch_size=config['patch_size'], scale=config['factor'])
test_dataset = TestDataset(config['valid_dataset_path'], scale=config['factor'])
ECE, absECE = test(model, test_dataset, patch_size=512, epsilon=epsilon)
if absECE > 0:
    start = 0.5
else:
    start = -0.5

lamda2 = start
deltamag = 0.1
lamda_list = []
absECE_list = []
loss_fn = loss.lossfun()
amp2,_ = finetune(model, loss_fn, train_dataset, test_dataset,
            epsilon=epsilon, alpha=0.1, lamda=lamda2)
lamda_list.append(lamda2)
absECE_list.append(amp2)

mag = lamda2 - deltamag * math.copysign(1, lamda2)
lamda3 = mag

model.module.load_state_dict(checkpt)
amp3,_ = finetune(model, loss_fn, train_dataset, test_dataset,
                epsilon=epsilon, alpha=0.1, lamda=lamda3)

if amp3 < amp2:
    while amp3 < amp2:
        amp1 = amp2
        lamda1 = lamda2
        amp2 = amp3
        lamda2 = lamda3
        mag -= deltamag * math.copysign(1, start)
        lamda3 = mag
        model.module.load_state_dict(checkpt)
        amp3,_ = finetune(model, loss_fn, train_dataset, test_dataset,
                epsilon=epsilon, alpha=0.1, lamda=lamda3)
else:
    mag = start
    a = amp3
    amp3 = amp2
    amp2 = a
    a = lamda3
    lamda3 = lamda2
    lamda2 = a
    while amp3 < amp2:
        amp1 = amp2
        lamda1 = lamda2
        amp2 = amp3
        lamda2 = lamda3
        mag += deltamag * math.copysign(1, start)
        lamda3 = mag
        model.module.load_state_dict(checkpt)
        amp3,_ = finetune(model, loss_fn, train_dataset, test_dataset,
                epsilon=epsilon, alpha=0.1, lamda=lamda3)

mag = fitxyparabola(lamda1, amp1, lamda2, amp2, lamda3, amp3)
print(mag)
model.module.load_state_dict(checkpt)
absECE, ECE = finetune(model, loss_fn, train_dataset, test_dataset, 
                epsilon=epsilon, alpha=0.1, lamda=mag)
torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + config['checkpoint_name'])


