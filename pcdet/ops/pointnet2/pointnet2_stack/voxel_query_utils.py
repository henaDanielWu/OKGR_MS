from typing import List
import torch
from . import pointnet2_stack_cuda as pointnet2
from . import pointnet2_utils
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

class VoxelQuery(x2ms_adapter.autograd.Function):

    @staticmethod
    def construct(ctx, max_range: int, radius: float, nsample: int, xyz, \
                    new_xyz, new_coords, point_indices):
        """
        Args:
            ctx:
            max_range: int, max range of voxels to be grouped
            nsample: int, maximum number of features in the balls
            new_coords: (M1 + M2, 4), [batch_id, z, y, x] cooridnates of keypoints
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            point_indices: (batch_size, Z, Y, X) 4-D tensor recording the point indices of voxels
        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert new_coords.is_contiguous()
        assert point_indices.is_contiguous()

        M = new_coords.shape[0]
        B, Z, Y, X = point_indices.shape
        idx = x2ms_adapter.tensor_api.zero_(x2ms_adapter.IntTensor(M, nsample))

        z_range, y_range, x_range = max_range
        pointnet2.voxel_query_wrapper(M, Z, Y, X, nsample, radius, z_range, y_range, x_range, \
                    new_xyz, xyz, new_coords, point_indices, idx)

        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

voxel_query = VoxelQuery.apply


class VoxelQueryAndGrouping(nn.Cell):
    def __init__(self, max_range: int, radius: float, nsample: int):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.max_range, self.radius, self.nsample = max_range, radius, nsample

    def construct(self, new_coords, xyz, xyz_batch_cnt,
                new_xyz, new_xyz_batch_cnt,
                features, voxel2point_indices):
        """
        Args:
            new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group
            voxel2point_indices: (B, Z, Y, X) tensor of points indices of voxels

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == x2ms_adapter.tensor_api.x2ms_sum(xyz_batch_cnt), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_coords.shape[0] == x2ms_adapter.tensor_api.x2ms_sum(new_xyz_batch_cnt), \
            'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
        batch_size = xyz_batch_cnt.shape[0]
        
        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx1, empty_ball_mask1 = voxel_query(self.max_range, self.radius, self.nsample, xyz, new_xyz, new_coords, voxel2point_indices)

        idx1 = x2ms_adapter.tensor_api.view(idx1, batch_size, -1, self.nsample)
        count = 0
        for bs_idx in range(batch_size):
            idx1[bs_idx] -= count
            count += xyz_batch_cnt[bs_idx]
        idx1 = x2ms_adapter.tensor_api.view(idx1, -1, self.nsample)
        idx1[empty_ball_mask1] = 0

        idx = idx1
        empty_ball_mask = empty_ball_mask1
        
        grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        # grouped_features: (M1 + M2, C, nsample)
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  
        
        return grouped_features, grouped_xyz, empty_ball_mask
