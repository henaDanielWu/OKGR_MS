import torch
from ...utils import common_utils
from . import roiaware_pool3d_cuda
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    print("rw.points_in_boxes_cpu")
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = x2ms_adapter.tensor_api.new_zeros(points, (boxes.shape[0], points.shape[0]), dtype=mindspore.int32)
    point_indices=torch.Tensor(x2ms_adapter.tensor_api.numpy(point_indices)).int()
    boxes=torch.Tensor(x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.x2ms_float(boxes)))
    points=torch.Tensor(x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.x2ms_float(points)))
    pt=roiaware_pool3d_cuda.points_in_boxes_cpu(boxes, points, point_indices)
    point_indices=pt.numpy()
    point_indices=mindspore.Tensor(point_indices)
    return x2ms_adapter.tensor_api.numpy(point_indices) if is_numpy else point_indices


def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    print("rw.points_in_boxes_gpu")
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = torch.zeros(batch_size, num_points, dtype=torch.int).fill_(-1)
    boxes=torch.Tensor(boxes.asnumpy())
    points=torch.Tensor(points.asnumpy())
    # roiaware_pool3d_cuda.points_in_boxes_gpu(boxes, points, box_idxs_of_pts)
    box_idxs_of_pts=mindspore.Tensor(box_idxs_of_pts.numpy())
    return box_idxs_of_pts


class RoIAwarePool3d(nn.Cell):
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def construct(self, rois, pts, pts_feature, pool_method='max'):
        print("rw.RoIAwarePool3d")
        assert pool_method in ['max', 'avg']
        return x2ms_adapter.nn_cell.apply(RoIAwarePool3dFunction, rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


class RoIAwarePool3dFunction(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        Args:
            ctx:
            rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
            pts: (npoints, 3)
            pts_feature: (npoints, C)
            out_size: int or tuple, like 7 or (7, 7, 7)
            max_pts_each_voxel:
            pool_method: 'max' or 'avg'

        Returns:
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        print("rw.RoIAwarePool3dFunction")
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = x2ms_adapter.tensor_api.new_zeros(pts_feature, (num_rois, out_x, out_y, out_z, num_channels))
        argmax = x2ms_adapter.tensor_api.new_zeros(pts_feature, (num_rois, out_x, out_y, out_z, num_channels), dtype=mindspore.int32)
        pts_idx_of_voxels = x2ms_adapter.tensor_api.new_zeros(pts_feature, (num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=mindspore.int32)

        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        # x2ms_adapter.forward(roiaware_pool3d_cuda, rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward

        grad_in = x2ms_adapter.tensor_api.new_zeros(grad_out, (num_pts, num_channels))
        # roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, x2ms_adapter.tensor_api.contiguous(grad_out), grad_in, pool_method)

        return None, None, grad_in, None, None, None


if __name__ == '__main__':
    pass
