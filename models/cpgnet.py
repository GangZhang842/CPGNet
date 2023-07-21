import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbone, bird_view, range_view
from networks.backbone import get_module
import pytorch_lib

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax

import yaml
import copy
import pdb


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = pytorch_lib.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / (pModel.Voxel.bev_shape[0])
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / (pModel.Voxel.bev_shape[1])
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / (pModel.Voxel.bev_shape[2])

        self.point_feat_out_channels = pModel.point_feat_out_channels
        self.stage_num = len(self.point_feat_out_channels)

        self.build_network()
        self.build_loss()

    def build_loss(self):
        self.criterion_seg_cate = None
        print("Loss mode: {}".format(self.pModel.loss_mode))
        if self.pModel.loss_mode == 'ce':
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == 'ohem':
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=3.0, ignore_index=0)
        else:
            raise Exception('loss_mode must in ["ce", "ohem"]')
    
    def build_network(self):
        # build cascaded network
        self.point_pre_list = nn.ModuleList()
        self.point_post_list = nn.ModuleList()
        self.bev_net_list = nn.ModuleList()
        self.rv_net_list = nn.ModuleList()
        self.bev_grid2point_list = nn.ModuleList()
        self.rv_grid2point_list = nn.ModuleList()
        self.pred_layer_list = nn.ModuleList()

        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point_list = self.pModel.BEVParam.bev_grid2point_list

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point_list = self.pModel.RVParam.rv_grid2point_list

        fusion_mode = self.pModel.fusion_mode

        # stage 0
        self.point_pre_list.append(backbone.PointNetStacker(7, bev_context_layer[0], pre_bn=True, stack_num=2))
        self.bev_net_list.append(bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers))
        self.rv_net_list.append(range_view.RVNet(rv_base_block, rv_context_layer, rv_layers))
        self.bev_grid2point_list.append(get_module(bev_grid2point_list[0], in_dim=self.bev_net_list[0].out_channels[-1]))
        self.rv_grid2point_list.append(get_module(rv_grid2point_list[0], in_dim=self.rv_net_list[0].out_channels[-1]))

        point_fusion_channels = (bev_context_layer[0], self.bev_net_list[0].out_channels[-1], self.rv_net_list[0].out_channels[-1])
        self.point_post_list.append(eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels[0]))

        self.pred_layer_list.append(backbone.PredBranch(self.point_feat_out_channels[0], self.pModel.class_num))

        # stage 1 ~ (stage_num - 1)
        for n in range(1, self.stage_num):
            stage_index = n - 1

            # define stage n network
            self.point_pre_list.append(backbone.PointNetStacker(self.point_feat_out_channels[stage_index], bev_context_layer[0], pre_bn=False, stack_num=1))
            self.bev_net_list.append(bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers))
            self.rv_net_list.append(range_view.RVNet(rv_base_block, rv_context_layer, rv_layers))
            self.bev_grid2point_list.append(get_module(bev_grid2point_list[n], in_dim=self.bev_net_list[n].out_channels[-1]))
            self.rv_grid2point_list.append(get_module(rv_grid2point_list[n], in_dim=self.rv_net_list[n].out_channels[-1]))

            point_fusion_channels_n = (self.point_feat_out_channels[stage_index], self.bev_net_list[n].out_channels[-1], self.rv_net_list[n].out_channels[-1])
            self.point_post_list.append(eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels_n, out_channel=self.point_feat_out_channels[n]))

            self.pred_layer_list.append(backbone.PredBranch(self.point_feat_out_channels[n], self.pModel.class_num))
    
    def stage_n_forward(self, point_feat, pcds_coord_wl, pcds_sphere_coord, stage_index=0):
        '''
        Input:
            point_feat (BS, C, N, 1)
            pcds_coord_wl (BS, N, 2, 1), 2 -> (x_quan, y_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            stage_index, type: int, means the (stage_index).{th} stage forward
        Output:
            point_feat_out (BS, C1, N, 1)
        '''
        point_feat_tmp = self.point_pre_list[stage_index](point_feat)

        #range-view
        rv_input = VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_sphere_coord, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        rv_feat_past, rv_feat = self.rv_net_list[stage_index](rv_input)
        point_rv_feat = self.rv_grid2point_list[stage_index](rv_feat, pcds_sphere_coord)

        #bird-view
        bev_input = VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord_wl, output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
        bev_feat_past, bev_feat = self.bev_net_list[stage_index](bev_input)
        point_bev_feat = self.bev_grid2point_list[stage_index](bev_feat, pcds_coord_wl)

        #merge multi-view
        if stage_index == 0:
            point_feat_out = self.point_post_list[stage_index](point_feat_tmp, point_bev_feat, point_rv_feat)
            return point_feat_out
        else:
            point_feat_out = self.point_post_list[stage_index](point_feat, point_bev_feat, point_rv_feat)
            return point_feat_out
    
    def forward_once(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            pcds_xyzi (BS, 10, N, 1), 10 -> (x, y, z, intensity, dist, diff_x, diff_y, diff_z, phi, theta)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls_list, list of pytorch tensors
        '''
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()
        pcds_feat_history = [pcds_xyzi]
        pred_cls_list = []
        # each stage forward
        for n in range(self.stage_num):
            pcds_feat_n = self.stage_n_forward(pcds_feat_history[n], pcds_coord_wl, pcds_sphere_coord, stage_index=n)
            pcds_feat_history.append(pcds_feat_n)

            pred_cls_n = self.pred_layer_list[n](pcds_feat_n).float() #(BS, class_num, N, 1)
            pred_cls_list.append(pred_cls_n)

        return pred_cls_list

    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        '''
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss

    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, 10, N, 1), 10 -> (x, y, z, intensity, dist, diff_x, diff_y, diff_z, phi, theta)
            pcds_coord, pcds_coord_raw (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        Output:
            loss_list (1, stage_num)
        '''
        pred_cls_list = self.forward_once(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        pred_cls_raw_list = self.forward_once(pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)

        loss_list = []
        # each stage loss function
        for n in range(len(pred_cls_list)):
            loss_tmp = self.criterion_seg_cate(pred_cls_list[n], pcds_target) + 2 * lovasz_softmax(pred_cls_list[n], pcds_target, ignore=0)
            loss_raw_tmp = self.criterion_seg_cate(pred_cls_raw_list[n], pcds_target) + 2 * lovasz_softmax(pred_cls_raw_list[n], pcds_target, ignore=0)
            loss_consist = self.consistency_loss_l1(pred_cls_list[n], pred_cls_raw_list[n])

            loss_total = 0.5 * (loss_tmp + loss_raw_tmp) + loss_consist
            loss_list.append(loss_total)

        loss_list = torch.stack(loss_list, dim=0).view(-1)
        return loss_list

    def infer_val(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target):
        '''
        Input:
            pcds_xyzi (BS, 10, N, 1), 10 -> (x, y, z, intensity, dist, diff_x, diff_y, diff_z, phi, theta)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        Output:
            pred_cls_list (BS, class_num, N, stage_num)
        '''
        pred_cls_list = self.forward_once(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        pred_cls_list = torch.cat(pred_cls_list, dim=-1)
        return pred_cls_list, pcds_target

    def infer_test(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            pcds_xyzi (BS, 10, N, 1), 10 -> (x, y, z, intensity, dist, diff_x, diff_y, diff_z, phi, theta)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls_list (BS, class_num, N, 1)
        '''
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()

        pcds_feat_history = [pcds_xyzi]
        # each stage forward
        for n in range(self.stage_num):
            pcds_feat_n = self.stage_n_forward(pcds_feat_history[n], pcds_coord_wl, pcds_sphere_coord, stage_index=n)
            pcds_feat_history.append(pcds_feat_n)

        pred_cls = self.pred_layer_list[-1](pcds_feat_history[-1])
        return pred_cls