# This file is modified from https://github.com/traveller59/second.pytorch

import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
try:
    from collections.abc import Iterable
except:
    from collections import Iterable

bn_types = (x2ms_nn.BatchNorm1d, x2ms_nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def split_bn_bias(layer_groups):
    "Split the layers in `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in x2ms_adapter.nn_cell.children(l):
            if isinstance(c, bn_types):
                l2.append(c)
            else:
                l1.append(c)
        split_groups += [x2ms_nn.Sequential(*l1).trainable_params(), x2ms_nn.Sequential(*l2).trainable_params()]
    return split_groups


def get_master(layer_groups, flat_master: bool = False):
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_groups = split_bn_bias(layer_groups)
    model_params = [[param for param in x2ms_adapter.parameters(lg) if param.requires_grad] for lg in split_groups]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) != 0:
                # mp = parameters_to_vector([x2ms_adapter.tensor_api.x2ms_float(param.data) for param in lg])
                mp = [x2ms_adapter.tensor_api.x2ms_float(param.data) for param in lg]
                mp = mindspore.Parameter(mp, requires_grad=True)
                if mp.grad is None: mp.grad = x2ms_adapter.tensor_api.new(mp, *x2ms_adapter.tensor_api.x2ms_size(mp))
                master_params.append([mp])
            else:
                master_params.append([])
        return model_params, master_params
    else:
        master_params = [[x2ms_adapter.tensor_api.detach(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.clone(param))) for param in lg] for lg in model_params]
        for mp in master_params:
            for param in mp: param.requires_grad = True
        return model_params, master_params


def model_g2master_g(model_params, master_params, flat_master: bool = False) -> None:
    "Copy the `model_params` gradients to `master_params` for the optimizer step."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(master_group) != 0:
                x2ms_adapter.tensor_api.copy_(master_group[0].grad.data, [x2ms_adapter.tensor_api.x2ms_float(p.grad.data) for p in model_group])
                # x2ms_adapter.tensor_api.copy_(master_group[0].grad.data, parameters_to_vector([x2ms_adapter.tensor_api.x2ms_float(p.grad.data) for p in model_group]))
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None: master.grad = x2ms_adapter.tensor_api.new(master.data, *x2ms_adapter.tensor_api.x2ms_size(master.data))
                    x2ms_adapter.tensor_api.copy_(master.grad.data, model.grad.data)
                else:
                    master.grad = None


# def master2model(model_params, master_params, flat_master: bool = False) -> None:
#     "Copy `master_params` to `model_params`."
#     if flat_master:
#         for model_group, master_group in zip(model_params, master_params):
#             if len(model_group) != 0:
#                 for model, master in zip(model_group, _unflatten_dense_tensors(master_group[0].data, model_group)):
#                     x2ms_adapter.tensor_api.copy_(model.data, master)
#     else:
#         for model_group, master_group in zip(model_params, master_params):
#             for model, master in zip(model_group, master_group): x2ms_adapter.tensor_api.copy_(model.data, master.data)


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def trainable_params(m: nn.Cell):
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, x2ms_adapter.parameters(m))
    return res


def is_tuple(x) -> bool: return isinstance(x, tuple)


# copy from fastai.
class OptimWrapper():
    "Basic wrapper around `opt` to simplify hyper-parameters changes."

    def __init__(self, opt, wd, true_wd: bool = False, bn_wd: bool = True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func, lr,param,
               layer_groups, **kwargs):
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        # split_groups = split_bn_bias(layer_groups)
#         conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
# no_conv_params = list(
#     filter(lambda x: 'conv' not in x.name, net.trainable_params()))
# group_params = [
#     {'params': conv_params, 'weight_decay': 0.01, 'lr': 0.9, "amsgrad": True},
#     {'params': no_conv_params, 'lr': 0.66, "eps": 1e-6, "betas": (0.8, 0.88)}]
# optimizer = optim.Adam(params=group_params, lr=0.01)
#         print(optimizer.param_groups)https://www.mindspore.cn/tutorials/zh-CN/r2.2/advanced/modules/optimizer.html?highlight=%E5%8A%A8%E6%80%81#%E5%AD%A6%E4%B9%A0%E7%8E%87
        opt = nn.Adam(params=param, learning_rate=0)
        opt = cls(opt, **kwargs)
        opt.lr, opt.opt_func = listify(lr, layer_groups), opt_func
        return opt

    def new(self, layer_groups):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr': 0} for l in split_groups])
        return self.create(opt_func, self.lr, layer_groups, wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def __repr__(self) -> str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    # Pytorch optimizer methods
    def step(self) -> None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(self._lr, self._wd, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
                for p in pg1['params']:
                    # When some parameters are fixed:  Shaoshuai Shi
                    if p.requires_grad is False:
                        continue
                    x2ms_adapter.tensor_api.mul_(p.data, 1 - wd * lr)
                if self.bn_wd:
                    for p in pg2['params']:
                        # When some parameters are fixed:  Shaoshuai Shi
                        if p.requires_grad is False:
                            continue
                        x2ms_adapter.tensor_api.mul_(p.data, 1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) -> None:
        "Clear optimizer gradients."
        x2ms_adapter.nn_cell.zero_grad(self.opt)

    # Passthrough to the inner opt.
    def __getattr__(self, k: str):
        return getattr(self.opt, k, None)

    def clear(self):
        "Reset the state of the inner optimizer."
        sd = x2ms_adapter.nn_cell.state_dict(self)
        sd['state'] = {}
        x2ms_adapter.load_state_dict(self, sd)

    # Hyperparameters as properties
    @property
    def lr(self) -> float:
        return self._lr[-1]

    @lr.setter
    def lr(self, val: float) -> None:
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self) -> float:
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) -> float:
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) -> float:
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        "Set weight decay."
        if not self.true_wd: self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    # Helper functions
    def read_defaults(self) -> None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')

    def set_val(self, key: str, val, bn_groups: bool = True):
        "Set `val` inside the optimizer dictionary at `key`."
        if is_tuple(val): val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key: str):
        "Read a hyperparameter `key` in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val


# class FastAIMixedOptim(OptimWrapper):
#     @classmethod
#     def create(cls, opt_func, lr,
#                layer_groups, model, flat_master=False, loss_scale=512.0, **kwargs):
#         "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
#         opt = OptimWrapper.create(opt_func, lr, layer_groups, **kwargs)
#         opt.model_params, opt.master_params = get_master(layer_groups, flat_master)
#         opt.flat_master = flat_master
#         opt.loss_scale = loss_scale
#         opt.model = model
#         # Changes the optimizer so that the optimization step is done in FP32.
#         # opt = self.learn.opt
#         mom, wd, beta = opt.mom, opt.wd, opt.beta
#         lrs = [lr for lr in opt._lr for _ in range(2)]
#         opt_params = [{'params': mp, 'lr': lr} for mp, lr in zip(opt.master_params, lrs)]
#         opt.opt = opt_func(opt_params)
#         opt.mom, opt.wd, opt.beta = mom, wd, beta
#         return opt
#
#     def step(self):
#         model_g2master_g(self.model_params, self.master_params, self.flat_master)
#         for group in self.master_params:
#             for param in group: x2ms_adapter.tensor_api.div_(param.grad, self.loss_scale)
#         super(FastAIMixedOptim, self).step()
#         x2ms_adapter.nn_cell.zero_grad(self.model)
#         # Update the params from master to model.
#         master2model(self.model_params, self.master_params, self.flat_master)
