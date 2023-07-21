import os

def get_config():
    class General:
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_size_per_gpu = 6
        fp16 = True

        SeqDir = './data/nuscenes'
        category_list = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']

        loss_mode = 'ohem'

        class Voxel:
            RV_theta = (-40.0, 20.0)
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-5.0, 3.0)

            bev_shape = (600, 600, 30)
            rv_shape = (64, 2048)

    class DatasetParam:
        class Train:
            data_src = 'data'
            num_workers = 4
            frame_point_num = 60000
            SeqDir = General.SeqDir
            fname_pkl = os.path.join(SeqDir, 'nuscenes_infos_train.pkl')
            Voxel = General.Voxel
            class AugParam:
                noise_mean = 0
                noise_std = 0.0001
                theta_range = (-180.0, 180.0)
                shift_range = ((-3, 3), (-3, 3), (-0.4, 0.4))
                size_range = (0.95, 1.05)

        class Val:
            data_src = 'data'
            num_workers = 4
            frame_point_num = None
            SeqDir = General.SeqDir
            fname_pkl = os.path.join(SeqDir, 'nuscenes_infos_val.pkl')
            Voxel = General.Voxel

    class ModelParam:
        prefix = "cpgnet"
        Voxel = General.Voxel
        category_list = General.category_list
        class_num = len(category_list) + 1
        loss_mode = General.loss_mode

        point_feat_out_channels = (64, 96)
        fusion_mode = 'CatFusion'

        class BEVParam:
            base_block = 'BasicBlock'
            context_layers = (64, 32, 64, 128)
            layers = (2, 3, 4)
            bev_grid2point_list = [
                dict(type='BilinearSample', scale_rate=(0.5, 0.5)),
                dict(type='BilinearSample', scale_rate=(0.5, 0.5))
            ]
        
        class RVParam:
            base_block = 'BasicBlock'
            context_layers = (64, 32, 64, 128)
            layers = (2, 3, 4)
            rv_grid2point_list = [
                dict(type='BilinearSample', scale_rate=(1.0, 0.5)),
                dict(type='BilinearSample', scale_rate=(1.0, 0.5))
            ]

        class pretrain:
            pretrain_epoch = 52

    class OptimizeParam:
        class optimizer:
            type = "sgd"
            base_lr = 0.02
            momentum = 0.9
            nesterov = True
            wd = 1e-3
        
        class schedule:
            type = "step"
            begin_epoch = 0
            end_epoch = 48
            pct_start = 0.01
            final_lr = 1e-6
            step = 10
            decay_factor = 0.1

    return General, DatasetParam, ModelParam, OptimizeParam