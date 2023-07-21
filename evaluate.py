import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True


def val_fp16(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate_list = []
    stage_num = model.stage_num
    for n in range(stage_num):
        criterion_cate_list.append(MultiClassMetric(category_list))
    print('FP16 inference mode!')
    model.eval()
    f = open(os.path.join(save_path, 'record_fp16_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, seq_id, fn) in tqdm.tqdm(enumerate(val_loader)):
            with torch.cuda.amp.autocast():
                pred_cls_list, pcds_target = model.infer_val(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(),\
                    pcds_sphere_coord.squeeze(0).cuda(), pcds_target.squeeze(0).cuda())
            
            pred_cls_list = F.softmax(pred_cls_list, dim=1)
            pred_cls_list = pred_cls_list.mean(dim=0).permute(2, 1, 0).contiguous()
            pcds_target = pcds_target[0].squeeze()
            for n in range(stage_num):
                criterion_cate_list[n].addBatch(pcds_target, pred_cls_list[n].contiguous())
        
        #record segmentation metric
        for n in range(stage_num):
            metric_cate = criterion_cate_list[n].get_metric()
            string = 'Epoch stage {0}: {1}'.format(n, epoch)
            for key in metric_cate:
                string = string + '; ' + key + ': ' + str(metric_cate[key])
            
            f.write(string + '\n')
        
        f.close()


def val(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate_list = []
    stage_num = model.stage_num
    for n in range(stage_num):
        criterion_cate_list.append(MultiClassMetric(category_list))
    
    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, seq_id, fn) in tqdm.tqdm(enumerate(val_loader)):
            pred_cls_list, pcds_target = model.infer_val(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(),\
                pcds_sphere_coord.squeeze(0).cuda(), pcds_target.squeeze(0).cuda())
            
            pred_cls_list = F.softmax(pred_cls_list, dim=1)
            pred_cls_list = pred_cls_list.mean(dim=0).permute(2, 1, 0).contiguous()
            pcds_target = pcds_target[0].squeeze()
            for n in range(stage_num):
                criterion_cate_list[n].addBatch(pcds_target, pred_cls_list[n].contiguous())
        
        #record segmentation metric
        for n in range(stage_num):
            metric_cate = criterion_cate_list[n].get_metric()
            string = 'Epoch stage {0}: {1}'.format(n, epoch)
            for key in metric_cate:
                string = string + '; ' + key + ': ' + str(metric_cate[key])
            
            f.write(string + '\n')
        
        f.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    # reset dist
    local_rank = int(os.getenv("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # define dataloader
    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)
    
    # define model
    model = eval(pModel.prefix).AttNet(pModel)
    model.cuda()
    model.eval()
    
    for epoch in range(args.start_epoch, args.end_epoch + 1, world_size):
        if (epoch + rank) < (args.end_epoch + 1):
            pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch + rank))
            model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            if pGen.fp16:
                val_fp16(epoch + rank, model, val_loader, pGen.category_list, save_path, rank)
            else:
                val(epoch + rank, model, val_loader, pGen.category_list, save_path, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)