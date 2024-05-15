import argparse
import yaml
import os
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import math
import random
import cv2
from PIL import Image
from VRT import VRT
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
num_dropout_ensembles = 8
parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = '3' # 0: 1:aug 2: ENsfinetune
#3:SISR 4 4:bigblur  5:non-neg 6:  7: 
#To do list: 2+3:lifeact 4+5
import numpy as np
from torch.utils.data import DataLoader
from dataset2 import TrainDataset
from dataset import TestDataset
import torch
import torch.nn as nn
import loss
from loss import FlowLoss
from mmedit.models.backbones.sr_backbones import Microvsr_net, BasicVSRPlusPlus, Resnet

#if config['use_wandb']:
 #   import wandb
  #  wandb.init(project=config['project'], entity=config['entity'], name=config['run_name'])

os.makedirs(config['checkpoint_folder'], exist_ok=True)
model = torch.nn.DataParallel(BasicVSRPlusPlus(propagation=config['propagation'],
                                             alignment=config['alignment'],
                                             bayes=config['bayes'])).cuda()
# model = torch.nn.DataParallel(VRT()).cuda()
# model = torch.nn.DataParallel(BasicVSRPlusPlus()).cuda()
# model = torch.nn.DataParallel(Microvsr_net(factor=config['factor'],
#                                            mid_channels=config['in_channels'], 
#                                            encoder_nblocks=config['encoder_nblocks'],
#                                            decoder_nblocks=config['decoder_nblocks'],
#                                            upsamle_nblocks=config['upsamle_nblocks'],
#                                            propagation=config['propagation'],
#                                            alignment=config['alignment'],
#                                            second_order=config['second_order'],
#                                            window_length=config['frame_length']),).cuda()
# model = torch.nn.DataParallel(Resnet.resnet34()).cuda()
#model = torch.nn.DataParallel(modules.SOFVSR(n_frames=config['frame_length'], is_training=True))
#model = models.mana(config,is_training=True).cuda()

if config['hot_start']:
    checkpt=torch.load(config['hot_start_checkpt'])
    model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

writer = SummaryWriter(log_dir=config['checkpoint_folder'])

train_dataset = TrainDataset(config['train_dataset_path'], patch_size=config['patch_size'], scale=config['factor'])
dataloader = DataLoader(dataset=train_dataset,
                        batch_size=config['batchsize'],
                        shuffle=True,
                        num_workers=config['num_workers'],
                        pin_memory=True)
test_dataset = TestDataset(config['valid_dataset_path'], patch_size=config['patch_size'], scale=config['factor'])
test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)
'''
test_dataset = TrainDataset('Micro_test.h5', patch_size=config['patch_size'], scale=3)
testloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)

'''
def calc_psnr(sr, hr, scale=3, rgb_range=255, dataset=None):
    #if hr.nelement() == 1: return 0
    #diff = (sr - hr) / rgb_range
    diff = (sr - hr)
    diff = diff.cpu().detach().numpy()
    mse = np.mean((diff) ** 2)
    #mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
count = 0
alpha = 0.5
stage1=config['stage1']
stage2=config['stage2']
stage3=6000
best_valid = [100,100,100,100]
optimizer = torch.optim.Adam(model.module.parameters(), lr=5e-5, betas=(0.5, 0.999))
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000], gamma=0.5)
for epoch in range(0, config['epoch']):
    loss_list=[]
    loss_list2=[]
    loss_list3=[]
    qloss_list=[]
    valid_list = []
    valid_flow_list = []
    model.train()
    with tqdm(dataloader, desc="Training MANA") as tepoch:
        for inp, gt in tepoch:
            tepoch.set_description(f"Training MANA--Epoch {epoch}")
            # for p in model.module.spynet.parameters():
            #     p.requires_grad=False
            #model.module.nonloc_spatial.mb.requires_grad=True
            #model.module.nonloc_spatial.D.requires_grad=False

            # gt = gt.reshape(-1, config['patch_size']*config['factor'], config['patch_size']*config['factor']).numpy()
            # gt = cv2.GaussianBlur(gt, (3,3), 0.8)
            # gt = gt.reshape(config['batchsize'], config['frame_length'], 1, config['patch_size']*config['factor'], config['patch_size']*config['factor'])
            # gt = torch.from_numpy(gt)
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            # gt_numpy = gt[1,0,0,::].data.cpu().numpy()
            # gt_numpy = Image.fromarray(gt_numpy)
            # gt_numpy.save('result/gt0.tif')
            # gt_numpy = gt[1,1,0,::].data.cpu().numpy()
            # gt_numpy = Image.fromarray(gt_numpy)
            # gt_numpy.save('result/gt1.tif')
            # gt_numpy = gt[1,2,0,::].data.cpu().numpy()
            # gt_numpy = Image.fromarray(gt_numpy)
            # gt_numpy.save('result/gt2.tif')
            # gt_numpy = inp[1,0,0,::].data.cpu().numpy()
            # gt_numpy = Image.fromarray(gt_numpy)
            # gt_numpy.save('result/inp0.tif')
            # gt_numpy = inp[1,1,0,::].data.cpu().numpy()
            # gt_numpy = Image.fromarray(gt_numpy)
            # gt_numpy.save('result/inp1.tif')
            # gt_numpy = inp[1,2,0,::].data.cpu().numpy()
            # gt_numpy = Image.fromarray(gt_numpy)
            # gt_numpy.save('result/inp2.tif')
            
            optimizer.zero_grad()
            # oup = model(inp)
            oup, flows_forward, flows_backward = model(inp)
            
            if count<stage3:
                wholeloss = loss_fn(gt[:,:,:,:,:], oup[:,:,:,:,:], config['bayes'])
                Warploss = FlowLoss(flows_forward, flows_backward, inp)
                loss = wholeloss
                l1loss = F.l1_loss(gt[:,:,0,:,:], oup[:,:,0,:,:])
                # loss = l1loss + alpha * Warploss
                #loss_numpy = loss.cpu().data.numpy()
                loss = loss.mean()
                loss.backward()
                loss_list.append(loss.data.cpu())
                loss_list2.append(Warploss.data.cpu())
                loss_list3.append(l1loss.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage1'})
                #if config['use_wandb']:
                 #   wandb.log({"L1 Loss": loss})
                
                    
            # elif count<stage2:
            #     loss=torch.mean(qloss)
            #     loss.backward()
            #     qloss_list.append(loss.data.cpu())
            #     loss1,loss2 = loss_fn(gt, oup)
            #     loss1 = loss1.mean()
            #     loss_list.append(loss1.data.cpu())
            #     loss_list2.append(loss2.data.cpu())
            #     optimizer.step()
            #     tepoch.set_postfix({'Quantize Loss:': loss1.data.cpu().numpy(),'L2 Loss': loss2.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage2'})
            #     #if config['use_wandb']:
            #      #   wandb.log({"Quantize Loss": loss})
                
            else:
                wholeloss = loss_fn(gt[:,:,:,:,:], oup[:,:,:,:,:], config['bayes'])
                Warploss = FlowLoss(flows_forward, flows_backward, inp)
                loss = wholeloss
                # loss = l1loss + alpha * Warploss
                loss = loss.mean()
                loss.backward()
                l1loss = F.l1_loss(gt[:,:,0,:,:], oup[:,:,0,:,:])
                loss_list.append(loss.data.cpu())
                loss_list2.append(Warploss.data.cpu())
                loss_list3.append(l1loss.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage3'})
                #if config['use_wandb']:
                  #  wandb.log({"L1 Loss": loss})

            count += 1
    
            if count % config['N_save_checkpt'] == 0:
                tepoch.set_description("Training MANA--Saving Checkpoint")
                torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + config['checkpoint_name'])
                with torch.no_grad():
                    model.eval()
                    with tqdm(test_dataloader, desc="Valid MANA") as tepoch:
                        for inp, gt in tepoch:
                            model.eval()
                            # gt = gt.reshape(-1, 192*config['factor'], 192*config['factor']).numpy()
                            # gt = cv2.GaussianBlur(gt, (3,3), 0.8)
                            # gt = gt.reshape(1, config['frame_length'], 1, 192*config['factor'], 192*config['factor'])
                            # gt = torch.from_numpy(gt)
                            inp = inp.float().cuda()
                            gt = gt.float().cuda()
                            optimizer.zero_grad()
                            oup, flows_forward, flows_backward = model(inp)
                            # oup = model(inp)
                            loss = loss_fn(gt[:,:,:,:,:], oup[:,:,:,:,:], config['bayes'])
                            # Flowloss = loss_fn(flows_forward, gtflow_forward) + loss_fn(flows_backward, gtflow_backward)
                            Warploss = FlowLoss(flows_forward, flows_backward, inp)
                            loss = loss.mean()
                            valid_list.append(loss.data.cpu())
                            valid_flow_list.append(Warploss.data.cpu())
                writer.add_scalar('Valid/loss', torch.mean(torch.stack(valid_list)), count / config['N_save_checkpt'])
                writer.add_scalar('Valid/Flowloss', torch.mean(torch.stack(valid_flow_list)), count / config['N_save_checkpt'])
                if count % (config['N_save_checkpt'] * 10) == 0:
                    writer.add_image('Valid/example', oup[0,0,0:1,::], count)
                if torch.mean(torch.stack(valid_list)) < max(best_valid):
                    idx = best_valid.index(max(best_valid))
                    best_valid[idx] = torch.mean(torch.stack(valid_list))
                    torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + 'checkptbest{}.pt'.format(idx))
                
    writer.add_scalar('Train/loss', torch.mean(torch.stack(loss_list3)), epoch)
    # if count < stage2 and count > stage1:
    #     writer.add_scalar('Train/qloss', torch.mean(torch.stack(qloss_list)), epoch)
    writer.add_scalar('Train/Flowloss', torch.mean(torch.stack(loss_list2)), epoch)
    writer.add_scalar('Train/wholeloss', torch.mean(torch.stack(loss_list)), epoch)
    scheduler.step()
    writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
writer.close()