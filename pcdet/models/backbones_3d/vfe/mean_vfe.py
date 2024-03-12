
import mindspore
import mindspore.nn as nn
import x2ms_adapter


class MeanVFE(nn.Cell):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__()
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def construct(self, batch_dict):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict.get('voxels'), batch_dict.get('voxel_num_points')
        points_mean = mindspore.ops.sum(voxel_features[:, :, :], dim=1, keepdim=False)
        # normalizer = x2ms_adapter.tensor_api.type_as(torch.clamp_min(x2ms_adapter.tensor_api.view(voxel_num_points, -1, 1), min=1.0), voxel_features)
        normalizer = mindspore.ops.clamp(x2ms_adapter.tensor_api.view(voxel_num_points, -1, 1), min=1.0)
        points_mean = points_mean / normalizer
        # batch_dict['voxel_features'] = points_mean
        batch_dict.update({'voxel_features': points_mean})

        return batch_dict

