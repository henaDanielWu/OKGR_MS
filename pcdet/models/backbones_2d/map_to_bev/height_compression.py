import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class HeightCompression(nn.Cell):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def construct(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']#是sptensor
        # spatial_features = encoded_spconv_tensor.dense()
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)
        # batch_dict['spatial_features'] = mindspore.Tensor(spatial_features.numpy())
        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
