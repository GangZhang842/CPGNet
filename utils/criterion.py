import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb

# define the online semi-hard examples mining binary cross entropy
class CE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0, weight=None, ignore_index=-1):
        super(CE_OHEM,self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        self.weight = weight
        self.ignore_index = ignore_index

        self.loss_func = nn.CrossEntropyLoss(weight=self.weight, reduce=False, ignore_index=self.ignore_index)
    
    def forward(self, pred, gt):
        #pdb.set_trace()
        loss_mat = self.loss_func(pred, gt.long())

        loss = loss_mat.view(1, -1)
        topk_num = max(int(self.top_ratio * loss.shape[1]), 1)
        loss_topk = torch.topk(loss, k=topk_num, dim=1, largest=True, sorted=False)[0]
        return loss.mean() + self.top_weight * loss_topk.mean()


# define the online semi-hard examples mining binary cross entropy
class BCE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0):
        super(BCE_OHEM,self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
    
    def forward(self, pred, gt):
        #pdb.set_trace()
        loss_mat = -1 * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12))

        loss = loss_mat.view(1, -1)
        topk_num = max(int(self.top_ratio * loss.shape[1]), 1)
        loss_topk = torch.topk(loss, k=topk_num, dim=1, largest=True, sorted=False)[0]
        return loss.mean() + self.top_weight * loss_topk.mean()