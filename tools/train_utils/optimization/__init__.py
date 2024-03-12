from functools import partial

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle
from x2ms_adapter.torch_api.optimizers import optim_register
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.lr_schedulers as lr_schedule_wrapper
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim_register.adam(x2ms_adapter.parameters(model), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim_register.sgd(
            x2ms_adapter.parameters(model), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Cell):
            return list(x2ms_adapter.nn_cell.children(m))

        def num_children(m: nn.Cell) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, x2ms_adapter.nn_cell.children(m)), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [x2ms_nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(nn.Adam, beta1=0.9, beta2=0.99)
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, model.trainable_params(),get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_schedule_wrapper.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
