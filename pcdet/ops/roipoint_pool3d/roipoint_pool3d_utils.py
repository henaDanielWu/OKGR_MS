import torch
from ...utils import box_utils
from . import roipoint_pool3d_cuda
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class RoIPointPool3d(nn.Cell):
    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def construct(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        print("rp.RoIPointPool3d")
        return x2ms_adapter.nn_cell.apply(
            RoIPointPool3dFunction, points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points
        )


class RoIPointPool3dFunction(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        print("rp.RoIPointPool3dFunction")
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[1], point_features.shape[2]
        pooled_boxes3d = x2ms_adapter.tensor_api.view(box_utils.enlarge_box3d(x2ms_adapter.tensor_api.view(boxes3d, -1, 7), pool_extra_width), batch_size, -1, 7)

        pooled_features = x2ms_adapter.tensor_api.new_zeros(point_features, (batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = x2ms_adapter.tensor_api.x2ms_int(x2ms_adapter.tensor_api.new_zeros(point_features, (batch_size, boxes_num)))

        x2ms_adapter.forward(
            roipoint_pool3d_cuda, x2ms_adapter.tensor_api.contiguous(points), x2ms_adapter.tensor_api.contiguous(pooled_boxes3d),
            x2ms_adapter.tensor_api.contiguous(point_features), pooled_features, pooled_empty_flag
        )

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


if __name__ == '__main__':
    pass
