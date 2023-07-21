import torch

from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np

import yaml
import random
import json
from . import utils
import os

import pickle as pkl
from nuscenes import NuScenes


def make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, Voxel):
    # make point feat
    x = pcds_xyzi[:, 0].copy()
    y = pcds_xyzi[:, 1].copy()
    z = pcds_xyzi[:, 2].copy()
    intensity = pcds_xyzi[:, 3].copy()

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12

    # grid diff
    diff_x = pcds_coord[:, 0] - np.floor(pcds_coord[:, 0])
    diff_y = pcds_coord[:, 1] - np.floor(pcds_coord[:, 1])
    diff_z = pcds_coord[:, 2] - np.floor(pcds_coord[:, 2])

    # sphere diff
    phi_range_radian = (-np.pi, np.pi)
    theta_range_radian = (Voxel.RV_theta[0] * np.pi / 180.0, Voxel.RV_theta[1] * np.pi / 180.0)

    phi = phi_range_radian[1] - np.arctan2(x, y)
    theta = theta_range_radian[1] - np.arcsin(z / dist)

    diff_phi = pcds_sphere_coord[:, 0] - np.floor(pcds_sphere_coord[:, 0])
    diff_theta = pcds_sphere_coord[:, 1] - np.floor(pcds_sphere_coord[:, 1])

    point_feat = np.stack((x, y, z, intensity, dist, diff_x, diff_y), axis=-1)
    return point_feat


# define the class of dataloader
class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.load(f)
        
        self.aug = utils.DataAugment(noise_mean=config.AugParam.noise_mean,
                        noise_std=config.AugParam.noise_std,
                        theta_range=config.AugParam.theta_range,
                        shift_range=config.AugParam.shift_range,
                        size_range=config.AugParam.size_range)

        self.aug_raw = utils.DataAugment(noise_mean=0,
                        noise_std=0,
                        theta_range=(0, 0),
                        shift_range=((0, 0), (0, 0), (0, 0)),
                        size_range=(1, 1))
        
        # add training data
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=config.SeqDir, verbose=True)
        with open(config.fname_pkl, 'rb') as f:
            data_infos = pkl.load(f)['infos']
            for info in data_infos:
                lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                fname_labels = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])

                fname_pcds = os.path.join(config.SeqDir, '{}/{}/{}'.format(*info['lidar_path'].split('/')[-3:]))
                self.flist.append((fname_pcds, fname_labels, info['lidar_path'], self.nusc.get('lidarseg', lidar_sd_token)['filename']))
        
        print('Training Samples: ', len(self.flist))

    def form_batch(self, pcds_total):
        #augment pcds
        pcds_total = self.aug(pcds_total)

        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_target = pcds_total[:, -1]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_target = torch.LongTensor(pcds_target.astype(np.long))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_target.unsqueeze(-1)

    def form_batch_raw(self, pcds_total):
        #augment pcds
        pcds_total = self.aug_raw(pcds_total)

        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_target = pcds_total[:, -1]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1)

    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id, fn = self.flist[index]

        #load point clouds and label file
        pcds = np.fromfile(fname_pcds, dtype=np.float32, count=-1).reshape((-1, 5))[:, :4]
        
        pcds_label_use = np.fromfile(fname_labels, dtype=np.uint8).reshape((-1))
        pcds_label_use = utils.relabel(pcds_label_use, self.task_cfg['learning_map'])
        
        # merge pcds and labels
        pcds_total = np.concatenate((pcds, pcds_label_use[:, np.newaxis]), axis=1)

        # resample
        choice = np.random.choice(pcds_total.shape[0], self.frame_point_num, replace=True)
        pcds_total = pcds_total[choice]

        # preprocess
        pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target = self.form_batch(pcds_total.copy())
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw = self.form_batch_raw(pcds_total.copy())
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, seq_id, fn

    def __len__(self):
        return len(self.flist)


# define the class of dataloader
class DataloadVal(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.load(f)

        self.nusc = NuScenes(version='v1.0-trainval', dataroot=config.SeqDir, verbose=True)
        with open(config.fname_pkl, 'rb') as f:
            data_infos = pkl.load(f)['infos']
            for info in data_infos:
                lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                fname_labels = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])

                fname_pcds = os.path.join(config.SeqDir, '{}/{}/{}'.format(*info['lidar_path'].split('/')[-3:]))
                self.flist.append((fname_pcds, fname_labels, info['lidar_path'], self.nusc.get('lidarseg', lidar_sd_token)['filename']))

        print('Validation Samples: ', len(self.flist))

    def form_batch(self, pcds_total):
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_target = pcds_total[:, -1]
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_target = torch.LongTensor(pcds_target.astype(np.long))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_target.unsqueeze(-1)

    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id, fn = self.flist[index]

        #load point clouds and label file
        pcds = np.fromfile(fname_pcds, dtype=np.float32, count=-1).reshape((-1, 5))[:, :4]
        
        pcds_label_use = np.fromfile(fname_labels, dtype=np.uint8).reshape((-1))
        pcds_label_use = utils.relabel(pcds_label_use, self.task_cfg['learning_map'])
        
        # merge pcds and labels
        pcds_total = np.concatenate((pcds, pcds_label_use[:, np.newaxis]), axis=1)

        # data aug
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_sphere_coord_list = []
        pcds_target_list = []
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pcds_tmp = pcds_total.copy()
                pcds_tmp[:, 0] *= x_sign
                pcds_tmp[:, 1] *= y_sign
                pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target = self.form_batch(pcds_tmp)

                pcds_xyzi_list.append(pcds_xyzi)
                pcds_coord_list.append(pcds_coord)
                pcds_sphere_coord_list.append(pcds_sphere_coord)
                pcds_target_list.append(pcds_target)
        
        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_sphere_coord = torch.stack(pcds_sphere_coord_list, dim=0)
        pcds_target = torch.stack(pcds_target_list, dim=0)
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, seq_id, fn

    def __len__(self):
        return len(self.flist)