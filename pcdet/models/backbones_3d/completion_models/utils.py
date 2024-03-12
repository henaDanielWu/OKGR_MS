import math
import torch
import numpy as np
# from pcdet.models.backbones_3d.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, \
#     gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

class Conv1d(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=mindspore.ops.relu):
        super(Conv1d, self).__init__()
        self.conv = x2ms_nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = x2ms_nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def construct(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=mindspore.ops.relu):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def construct(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(xyz, 0, 2, 1)) # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    #采样完了的idx不对劲
    idx = ball_query(radius, nsample, xyz_flipped, x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(new_xyz, 0, 2, 1))) # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= x2ms_adapter.tensor_api.repeat(x2ms_adapter.tensor_api.unsqueeze(new_xyz, 3), 1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = x2ms_adapter.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz





'''
# ------------test sample_and_group-------------
xyz = torch.randn((10, 3, 100), device='cuda:0')
points = torch.randn((10, 32, 100), device='cuda:0')

new_xyz, new_points, idx_local, grouped_xyz = sample_and_group(xyz, points, 10, 5, .7, use_xyz=False)
print('not use_xyz', new_xyz.shape, new_points.shape, idx_local.shape, grouped_xyz.shape)

new_xyz, new_points, idx_local, grouped_xyz = sample_and_group(xyz, points, 10, 5, .7)
print('use_xyz', new_xyz.shape, new_points.shape, idx_local.shape, grouped_xyz.shape)

new_xyz, new_points, idx_local, grouped_xyz = sample_and_group(xyz, None, 10, 5, .7)
print('points None', new_xyz.shape, new_points.shape, idx_local.shape, grouped_xyz.shape)
# ---------------------------------------------------
'''


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    new_xyz = x2ms_adapter.tensor_api.repeat(x2ms_adapter.zeros((1, 3, 1), dtype=mindspore.float32), b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = x2ms_adapter.tensor_api.repeat(x2ms_adapter.arange(nsample).reshape(1, 1, nsample), b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = x2ms_adapter.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = x2ms_adapter.tensor_api.unsqueeze(new_points, 2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

'''
# ------------test sample_and_group_all-------------
xyz = torch.randn((10, 3, 100), device='cuda:0')
points = torch.randn((10, 32, 100), device='cuda:0')

new_xyz, new_points, idx_local, grouped_xyz = sample_and_group_all(xyz, points, use_xyz=False)
print('not use_xyz', new_xyz.shape, new_points.shape, idx_local.shape, grouped_xyz.shape)

new_xyz, new_points, idx_local, grouped_xyz = sample_and_group_all(xyz, points)
print('use_xyz', new_xyz.shape, new_points.shape, idx_local.shape, grouped_xyz.shape)

new_xyz, new_points, idx_local, grouped_xyz = sample_and_group_all(xyz, None)
print('points None', new_xyz.shape, new_points.shape, idx_local.shape, grouped_xyz.shape)
# ---------------------------------------------------
'''

class PointNet_SA_Module(nn.Cell):
    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = x2ms_nn.Sequential(*self.mlp_conv)

    def construct(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample, self.radius, self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = x2ms_adapter.x2ms_max(new_points, 3)[0]

        return new_xyz, new_points


class PointNet_FP_Module(nn.Cell):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = x2ms_nn.Sequential(*self.mlp_conv)

    def construct(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1))
        dist=torch.tensor(dist.asnumpy())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0/dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        weight=mindspore.tensor(weight.numpy())
        interpolated_points = three_interpolate(points2, idx, weight) # B, in_channel, N

        if self.use_points1:
            new_points = x2ms_adapter.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


class PointNet_FP_Module2(nn.Cell):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module2, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = x2ms_nn.Sequential(*self.mlp_conv)

    def construct(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(xyz1, 0, 2, 1)), x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(xyz2, 0, 2, 1)))
        # dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        dist = mindspore.ops.clamp(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0/dist
        norm = x2ms_adapter.tensor_api.repeat(x2ms_adapter.x2ms_sum(recip_dist, 2, keepdim=True), (1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight) # B, in_channel, N

        if self.use_points1:
            new_points = x2ms_adapter.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points

'''
# -------test FP Module -------------

F = PointNet_FP_Module(32, [64, 128, 256]).cuda('cuda:0')
xyz1 = torch.randn((10, 3, 100), device='cuda:0')
xyz2 = torch.randn((10, 3, 50), device='cuda:0')
points1 = torch.randn((10, 32, 100), device='cuda:0')
points2 = torch.randn((10, 32, 50), device='cuda:0')
new_points = F(xyz1, xyz2, points1, points2)
print(new_points.shape)

F = PointNet_FP_Module(32, [64, 128, 256], use_points1=True, in_channel_points1=32).cuda('cuda:0')
xyz1 = torch.randn((10, 3, 100), device='cuda:0')
xyz2 = torch.randn((10, 3, 50), device='cuda:0')
points1 = torch.randn((10, 32, 100), device='cuda:0')
points2 = torch.randn((10, 32, 50), device='cuda:0')
new_points = F(xyz1, xyz2, points1, points2)
print(new_points.shape)


F = PointNet_FP_Module(64, [64, 128, 256], use_points1=True, in_channel_points1=32).cuda('cuda:0')
xyz1 = torch.randn((10, 3, 100), device='cuda:0')
xyz2 = torch.randn((10, 3, 50), device='cuda:0')
points1 = torch.randn((10, 32, 100), device='cuda:0')
points2 = torch.randn((10, 64, 50), device='cuda:0')
new_points = F(xyz1, xyz2, points1, points2)
print(new_points.shape)
# ---------------------------------------------------
'''



class MLP(nn.Cell):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(x2ms_nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(x2ms_nn.BatchNorm1d(out_channel))
            layers.append(x2ms_nn.ReLU())
            last_channel = out_channel
        layers.append(x2ms_nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = x2ms_nn.Sequential(*layers)

    def construct(self, inputs):
        return self.mlp(inputs)


class MLP_CONV(nn.Cell):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(x2ms_nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(x2ms_nn.BatchNorm1d(out_channel))
            layers.append(x2ms_nn.ReLU())
            last_channel = out_channel
        layers.append(x2ms_nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = x2ms_nn.Sequential(*layers)

    def construct(self, inputs):
        return self.mlp(inputs)

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(pcd, 0, 2, 1)), furthest_point_sample(pcd, n_points))
    new_pcd = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(new_pcd, 0, 2, 1))
    return new_pcd

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * x2ms_adapter.matmul(src, x2ms_adapter.tensor_api.permute(dst, 0, 2, 1))  # B, N, M
    dist += x2ms_adapter.tensor_api.view(x2ms_adapter.x2ms_sum(src ** 2, -1), B, N, 1)
    dist += x2ms_adapter.tensor_api.view(x2ms_adapter.x2ms_sum(dst ** 2, -1), B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = x2ms_adapter.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return x2ms_adapter.tensor_api.x2ms_int(idx)






















