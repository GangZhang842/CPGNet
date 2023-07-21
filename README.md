# CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation

Official code for [CPGNet](https://arxiv.org/abs/2204.09914)

> [**CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation**](https://arxiv.org/abs/2204.09914),
> Xiaoyan Li, Gang Zhang, Hongyu Pan, Zhenhua Wang.
> *Accepted by ICRA2022 ([arXiv:2204.09914](https://arxiv.org/abs/2204.09914))*


## NEWS
[2022-02-01] CPGNet is accepted by ICRA 2022


## 1 Dependency
```bash
CUDA>=10.2
Pytorch>=1.8.0
PyYAML@5.4.1
scipy@1.3.1
nuscenes
```

## 2 Training Process
### 2.1 Installation
```bash
python3 setup.py install
```

### 2.2 Training Script
```bash
torchrun --master_port=12098 --nproc_per_node=4 train.py --config config/config_cpgnet_sgd_bili_sample_ohem_fp16.py
```

## 3 Evaluate Process
```bash
torchrun --master_port=12097 --nproc_per_node=4 evaluate.py --config config/config_cpgnet_sgd_bili_sample_ohem_fp16.py --start_epoch 0 --end_epoch 47
```

## Citations
```bash
@inproceedings{li2022cpgnet,
  author={Li, Xiaoyan and Zhang, Gang and Pan, Hongyu and Wang, Zhenhua},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation}, 
  year={2022},
  volume={},
  number={},
  pages={11117-11123},
  doi={10.1109/ICRA46639.2022.9811767}
}
```