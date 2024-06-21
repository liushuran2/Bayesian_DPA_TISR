# By Shuran Liu, 2023
import argparse
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import math
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)

from torch.utils.data import DataLoader
from Traindataset import TrainDataset
from Testdataset import TestDataset
import torch
import loss
from mmedit.models.backbones.sr_backbones import DPATISR


os.makedirs(config['checkpoint_folder'], exist_ok=True)
os.makedirs(config['tensorboard_folder'], exist_ok=True)
model = torch.nn.DataParallel(DPATISR(mid_channels=config['mid_channels'],
                 extraction_nblocks=config['extraction_nblocks'],
                 propagation_nblocks=config['propagation_nblocks'],
                 reconstruction_nblocks=config['reconstruction_nblocks'],
                 factor=config['factor'],
                 bayesian=config['bayesian'])).cuda()

if config['hot_start']:
    checkpt=torch.load(config['hot_start_checkpt'])
    model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

writer = SummaryWriter(log_dir=config['tensorboard_folder'])

train_dataset = TrainDataset(config['train_dataset_path'], patch_size=config['patch_size'], scale=config['factor'])
dataloader = DataLoader(dataset=train_dataset,
                        batch_size=config['batchsize'],
                        shuffle=True,
                        num_workers=config['num_workers'],
                        pin_memory=True)
test_dataset = TestDataset(config['valid_dataset_path'], scale=config['factor'])
test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)
count = 0
best_valid = [100,100,100,100]
optimizer = torch.optim.Adam(model.module.parameters(), lr=5e-5, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000], gamma=0.5)
for epoch in range(0, config['epoch']):
    loss_list=[]
    loss_list_L1=[]
    valid_list = []
    model.train()
    with tqdm(dataloader, desc="Training") as tepoch:
        for inp, gt in tepoch:
            tepoch.set_description(f"Training--Epoch {epoch}")
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            
            optimizer.zero_grad()
            oup = model(inp)
            
            loss = loss_fn(gt[:,:,:,:,:], oup[:,:,:,:,:], config['bayesian'])
            l1loss = F.l1_loss(gt[:,:,0,:,:], oup[:,:,0,:,:])
            loss = loss.mean()
            loss.backward()
            loss_list.append(loss.data.cpu())
            loss_list_L1.append(l1loss.data.cpu())
            optimizer.step()
            tepoch.set_postfix({'loss': loss.data.cpu().numpy(), 'Current Iteration': count})

            count += 1
    
            if count % config['N_save_checkpt'] == 0:
                tepoch.set_description("Training--Saving Checkpoint")
                torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + config['checkpoint_name'])
                with torch.no_grad():
                    model.eval()
                    with tqdm(test_dataloader, desc="Validation") as tepoch:
                        for inp, gt in tepoch:
                            model.eval()
                            inp = inp.float().cuda()
                            gt = gt.float().cuda()
                            optimizer.zero_grad()
                            oup = model(inp)
                            loss = loss_fn(gt[:,:,:,:,:], oup[:,:,:,:,:], config['bayesian'])
                            loss = loss.mean()
                            valid_list.append(loss.data.cpu())
                writer.add_scalar('Valid/loss', torch.mean(torch.stack(valid_list)), count / config['N_save_checkpt'])
                if torch.mean(torch.stack(valid_list)) < max(best_valid):
                    idx = best_valid.index(max(best_valid))
                    best_valid[idx] = torch.mean(torch.stack(valid_list))
                    torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + 'checkptbest{}.pt'.format(idx))
                
    writer.add_scalar('Train/L1loss', torch.mean(torch.stack(loss_list_L1)), epoch)
    writer.add_scalar('Train/loss', torch.mean(torch.stack(loss_list)), epoch)
    scheduler.step()
    writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
writer.close()