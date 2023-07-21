import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import collections

from torch.optim import SGD, AdamW, lr_scheduler

from functools import partial
import math

import pdb


def schedule_with_warmup(k, num_epoch, per_epoch_num_iters, pct_start, step, decay_factor):
    warmup_iters = int(num_epoch * per_epoch_num_iters * pct_start)
    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        epoch = k // per_epoch_num_iters
        step_idx = (epoch // step)
        return math.pow(decay_factor, step_idx)


def get_scheduler(optimizer, pOpt, per_epoch_num_iters):
    num_epoch = pOpt.schedule.end_epoch - pOpt.schedule.begin_epoch
    if pOpt.schedule.type == 'OneCycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=pOpt.optimizer.base_lr,
                                            epochs=num_epoch, steps_per_epoch=per_epoch_num_iters,
                                            pct_start=pOpt.schedule.pct_start, anneal_strategy='cos',
                                            div_factor=25, final_div_factor=pOpt.optimizer.base_lr / pOpt.schedule.final_lr)
        return scheduler
    elif pOpt.schedule.type == 'step':
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                schedule_with_warmup,
                num_epoch=num_epoch,
                per_epoch_num_iters=per_epoch_num_iters,
                pct_start=pOpt.schedule.pct_start,
                step=pOpt.schedule.step,
                decay_factor=pOpt.schedule.decay_factor
            ))
        return scheduler
    else:
        raise NotImplementedError(pOpt.schedule.type)


def get_optimizer(pOpt, model):
    if pOpt.optimizer.type in ['adam', 'adamw']:
        optimizer = AdamW(params=model.parameters(),
                        lr=pOpt.optimizer.base_lr,
                        weight_decay=pOpt.optimizer.wd)
        return optimizer
    elif pOpt.optimizer.type == 'sgd':
        optimizer = SGD(params=model.parameters(),
                        lr=pOpt.optimizer.base_lr,
                        momentum=pOpt.optimizer.momentum,
                        weight_decay=pOpt.optimizer.wd,
                        nesterov=pOpt.optimizer.nesterov)
        return optimizer
    else:
        raise NotImplementedError(pOpt.optimizer.type)