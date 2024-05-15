# By Shuran Liu, 2023
import torch
import torch.nn.functional as F


class lossfun(torch.nn.Module):
    def __init__(self):
        super(lossfun, self).__init__()

    def forward(self, gt, oups, bayesian=False):
        if bayesian:
            mean = oups[:,:,0:1,:,:]
            std = oups[:,:,1:2,:,:]
            loss = torch.div(torch.abs(gt[:,:,:,:,:] - mean[:,:,:,:,:]), std + 1e-6) + torch.log(std + 1e-6)
        else:
            loss = F.l1_loss(oups, gt)
        return loss
    

