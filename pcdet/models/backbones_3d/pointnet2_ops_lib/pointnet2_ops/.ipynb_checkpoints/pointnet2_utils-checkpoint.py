from typing import Tuple

from . import pointnet2_batch_cuda as pointnet2
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class FarthestPointSampling(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, xyz, npoint: int) -> mindspore.Tensor:
        """
        Uses iterative farthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = x2ms_adapter.tensor_api.x2ms_size(xyz)
        output = x2ms_adapter.IntTensor(B, npoint)
        temp = x2ms_adapter.tensor_api.fill_(x2ms_adapter.FloatTensor(B, N), 1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply


class GatherOperation(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, features, idx) -> mindspore.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = x2ms_adapter.tensor_api.x2ms_size(idx)
        _, C, N = x2ms_adapter.tensor_api.x2ms_size(features)
        output = x2ms_adapter.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = x2ms_adapter.tensor_api.x2ms_size(idx)

        grad_features = x2ms_adapter.autograd.Variable(x2ms_adapter.tensor_api.zero_(x2ms_adapter.FloatTensor(B, C, N)))
        grad_out_data = x2ms_adapter.tensor_api.contiguous(grad_out.data)
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, unknown, known) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = x2ms_adapter.tensor_api.x2ms_size(unknown)
        m = x2ms_adapter.tensor_api.x2ms_size(known, 1)
        dist2 = x2ms_adapter.FloatTensor(B, N, 3)
        idx = x2ms_adapter.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return x2ms_adapter.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, features, idx, weight) -> mindspore.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = x2ms_adapter.tensor_api.x2ms_size(features)
        n = x2ms_adapter.tensor_api.x2ms_size(idx, 1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = x2ms_adapter.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = x2ms_adapter.tensor_api.x2ms_size(grad_out)

        grad_features = x2ms_adapter.autograd.Variable(x2ms_adapter.tensor_api.zero_(x2ms_adapter.FloatTensor(B, c, m)))
        grad_out_data = x2ms_adapter.tensor_api.contiguous(grad_out.data)

        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, features, idx) -> mindspore.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = x2ms_adapter.tensor_api.x2ms_size(idx)
        _, C, N = x2ms_adapter.tensor_api.x2ms_size(features)
        output = x2ms_adapter.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = x2ms_adapter.tensor_api.x2ms_size(grad_out)
        grad_features = x2ms_adapter.autograd.Variable(x2ms_adapter.tensor_api.zero_(x2ms_adapter.FloatTensor(B, C, N)))

        grad_out_data = x2ms_adapter.tensor_api.contiguous(grad_out.data)
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, radius: float, nsample: int, xyz, new_xyz) -> mindspore.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = x2ms_adapter.tensor_api.x2ms_size(xyz)
        npoint = x2ms_adapter.tensor_api.x2ms_size(new_xyz, 1)
        idx = x2ms_adapter.tensor_api.zero_(x2ms_adapter.IntTensor(B, npoint, nsample))

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Cell):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def construct(self, xyz, new_xyz, features = None) -> Tuple[mindspore.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.transpose(xyz, 1, 2))
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.transpose(new_xyz, 1, 2), -1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = x2ms_adapter.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Cell):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def construct(self, xyz, new_xyz, features = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.transpose(xyz, 1, 2), 2)
        if features is not None:
            grouped_features = x2ms_adapter.tensor_api.unsqueeze(features, 2)
            if self.use_xyz:
                new_features = x2ms_adapter.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
