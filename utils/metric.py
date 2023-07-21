import numpy as np
import torch
import collections
import pdb


class MultiClassMetric:
    def __init__(self, Classes):
        #0: background; 255: ignore labels
        self.Classes = Classes
        self.reset()
    
    def reset(self):
        self.tp = np.zeros(len(self.Classes), dtype=np.float32)
        self.pred_num = np.zeros(len(self.Classes), dtype=np.float32)
        self.gt_num = np.zeros(len(self.Classes), dtype=np.float32)
    
    def addBatch(self, gt, pred):
        #gt (bs, )
        #pred (bs, c); c denotes number of categories
        gt = gt.cpu()
        pred = pred.cpu()
        
        valid_mask = ((gt != 0).float()).data.cpu().numpy()
        _, pred_map = torch.max(pred, dim=1)
        
        gt_map = (gt.float()).data.cpu().numpy()
        pred_map = (pred_map.float()).data.cpu().numpy()
        
        gt_map[valid_mask == 0] = -1
        pred_map[valid_mask == 0] = -1
        for i, cate in enumerate(self.Classes):
            pred_tmp = (pred_map == (i + 1)).astype(np.float32)
            gt_tmp = (gt_map == (i + 1)).astype(np.float32)
            
            #pdb.set_trace()
            tp = (pred_tmp * gt_tmp).sum()
            pred_num = pred_tmp.sum()
            gt_num = gt_tmp.sum()
            
            self.tp[i] = self.tp[i] + tp
            self.pred_num[i] = self.pred_num[i] + pred_num
            self.gt_num[i] = self.gt_num[i] + gt_num
    
    def get_metric(self):
        result_dic = collections.OrderedDict()
        iou = self.tp / (self.gt_num + self.pred_num - self.tp + 1e-12)
        pre = self.tp / (self.pred_num + 1e-12)
        rec = self.tp / (self.gt_num + 1e-12)
        
        for i, cate in enumerate(self.Classes):
            result_dic[cate + ' iou'] = iou[i]
            result_dic[cate + ' pre'] = pre[i]
            result_dic[cate + ' rec'] = rec[i]
        
        result_dic['mean iou'] = iou.mean()
        self.reset()
        return result_dic
