# from functools import partial
#
# from ...utils.spconv_utils import replace_feature, spconv
# import mindspore
# import mindspore.nn as nn
# import x2ms_adapter
# import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
# import torch
# # import torch.nn
#
#
# def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
#                    conv_type='subm', norm_fn=None):
#
#     if conv_type == 'subm':
#         # conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
#         conv = nn.Conv3d(in_channels, out_channels, kernel_size, has_bias=False,pad_mode='pad')
#     elif conv_type == 'spconv':
#         # conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#         #                            bias=False, indice_key=indice_key)
#         conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                      has_bias=False,pad_mode='pad')
#     # elif conv_type == 'inverseconv':
#     #     conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
#     else:
#         raise NotImplementedError
#
#     # m = spconv.SparseSequential(
#     #     conv,
#     #     n=norm_fn(out_channels),
#     #     r=nn.ReLU()
#     # )
#     m = nn.SequentialCell(
#         conv,
#         norm_fn(out_channels),
#         nn.ReLU()
#     )
#     return m
#
#
# class SparseBasicBlock(spconv.SparseModule):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
#         super(SparseBasicBlock, self).__init__()
#
#         assert norm_fn is not None
#         bias = norm_fn is not None
#         self.conv1 = spconv.SubMConv3d(
#             inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
#         )
#         self.bn1 = norm_fn(planes)
#         self.relu = x2ms_nn.ReLU()
#         self.conv2 = spconv.SubMConv3d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
#         )
#         self.bn2 = norm_fn(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def construct(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = replace_feature(out, self.bn1(out.features))
#         out = replace_feature(out, self.relu(out.features))
#
#         out = self.conv2(out)
#         out = replace_feature(out, self.bn2(out.features))
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = replace_feature(out, out.features + identity.features)
#         out = replace_feature(out, self.relu(out.features))
#
#         return out
#
# # def scatter_nd(indices, updates, shape):
# #     """pytorch edition of tensorflow scatter_nd.
# #     this function don't contain except handle code. so use this carefully
# #     when indice repeats, don't support repeat add which is supported
# #     in tensorflow.
# #     """
# #     ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
# #     ndim = indices.shape[-1]
# #     output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
# #     flatted_indices = indices.view(-1, ndim)
# #     slices = [flatted_indices[:, i] for i in range(ndim)]
# #     slices += [Ellipsis]
# #     ret[slices] = updates.view(*output_shape)
# #     return ret
# # def dense(indices, batch_size,spatial_shape,features,channels_first: bool = True):
# #     output_shape = [batch_size] + list(
# #         spatial_shape) + [features.shape[1]]
# #     res = scatter_nd(
# #         indices.to(features.device).long(), features,
# #         output_shape)
# #     if not channels_first:
# #         return res
# #     ndim = len(spatial_shape)
# #     trans_params = list(range(0, ndim + 1))
# #     trans_params.insert(1, ndim + 1)
# #     return res.permute(*trans_params)
# class VoxelBackBone8x(nn.Cell):
#     def __init__(self, input_channels, grid_size, **kwargs):
#         super().__init__()
#         norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#
#         self.sparse_shape = grid_size[::-1] + [1, 0, 0]
#
#         # self.conv_input = spconv.SparseSequential(
#         #     spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#         #     norm_fn(16),
#         #     torch.nn.ReLU(),
#         # )
#         self.conv_input = nn.SequentialCell(
#             nn.Conv3d(input_channels, 16, 3, padding=1, has_bias=False,pad_mode='pad'),
#             norm_fn(16),
#             nn.ReLU(),
#         )
#         block = post_act_block
#
#         self.conv1 = nn.SequentialCell(
#             block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
#         )
#
#         self.conv2 = nn.SequentialCell(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#             block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
#         )
#
#         self.conv3 = nn.SequentialCell(
#             # [800, 704, 21] <- [400, 352, 11]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
#         )
#
#         self.conv4 = nn.SequentialCell(
#             # [400, 352, 11] <- [200, 176, 5]
#             block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1,0,0,0), indice_key='spconv4', conv_type='spconv'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#             block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
#         )
#
#
#         last_pad = 0
#         # last_pad = self.model_cfg.get('last_pad', last_pad)
#         self.conv_out = nn.SequentialCell(
#             # [200, 150, 5] -> [200, 150, 2]
#             nn.Conv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
#                                 has_bias=False,pad_mode='pad'),
#             norm_fn(128),
#             nn.ReLU(),
#         )
#
#         self.num_point_features = 128
#         self.backbone_channels = {
#             'x_conv1': 16,
#             'x_conv2': 32,
#             'x_conv3': 64,
#             'x_conv4': 64
#         }
#
#
#
#     def construct(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (num_voxels, C)
#                 voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         """
#         voxel_features, voxel_coords = batch_dict.get('voxel_features'), batch_dict.get('voxel_coords')
#         batch_size = batch_dict.get('batch_size')
#         voxel_features=torch.Tensor(voxel_features.asnumpy())
#         voxel_coords=torch.Tensor(voxel_coords.asnumpy()).int()
#         # batch_size=torch.Tensor(x2ms_adapter.tensor_api.numpy(batch_size))
#         # self.sparse_shape=torch.Tensor(x2ms_adapter.tensor_api.numpy(self.sparse_shape))
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=voxel_features,
#             indices=voxel_coords,
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )
#         input_sp_tensor=input_sp_tensor.dense()
#         input_sp_tensor=mindspore.Tensor(input_sp_tensor.numpy())
#         x = self.conv_input(input_sp_tensor)
#
#         x_conv1 = self.conv1(x)
#         x_conv2 = self.conv2(x_conv1)
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)
#
#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(x_conv4)
#         batch_dict.update({
#             'encoded_spconv_tensor': out,
#             'encoded_spconv_tensor_stride': 8
#         })
#
#         N, C, D, H, W = out.shape
#         spatial_features = out.view(N, C * D, H, W)
#         batch_dict.update({
#             'spatial_features':spatial_features
#         })
#         # batch_dict['spatial_features'] = spatial_features
#         # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
#         batch_dict.update({'spatial_features_stride': 8})
#         # x_conv1=mindspore.Tensor(x_conv1.numpy())
#         # x_conv2=mindspore.Tensor(x_conv2.numpy())
#         # x_conv3=mindspore.Tensor(x_conv3.numpy())
#         # x_conv4=mindspore.Tensor(x_conv4.numpy())
#         batch_dict.update({
#             'multi_scale_3d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#             }
#         })
#         batch_dict.update({
#             'multi_scale_3d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#             }
#         })
#         return batch_dict
#
#
# class VoxelResBackBone8x(nn.Cell):
#     def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         norm_fn = partial(x2ms_nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#
#         self.sparse_shape = grid_size[::-1] + [1, 0, 0]
#
#         self.conv_input = spconv.SparseSequential(
#             spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#             norm_fn(16),
#             x2ms_nn.ReLU(),
#         )
#         block = post_act_block
#
#         self.conv1 = spconv.SparseSequential(
#             SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
#             SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
#         )
#
#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
#         )
#
#         self.conv3 = spconv.SparseSequential(
#             # [800, 704, 21] <- [400, 352, 11]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
#         )
#
#         self.conv4 = spconv.SparseSequential(
#             # [400, 352, 11] <- [200, 176, 5]
#             block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
#         )
#
#         last_pad = 0
#         last_pad = self.model_cfg.get('last_pad', last_pad)
#         self.conv_out = spconv.SparseSequential(
#             # [200, 150, 5] -> [200, 150, 2]
#             spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
#                                 bias=False, indice_key='spconv_down2'),
#             norm_fn(128),
#             x2ms_nn.ReLU(),
#         )
#         self.num_point_features = 128
#         self.backbone_channels = {
#             'x_conv1': 16,
#             'x_conv2': 32,
#             'x_conv3': 64,
#             'x_conv4': 128
#         }
#
#     def construct(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (num_voxels, C)
#                 voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         """
#         voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
#         batch_size = batch_dict['batch_size']
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=voxel_features,
#             indices=x2ms_adapter.tensor_api.x2ms_int(voxel_coords),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )
#         x = self.conv_input(input_sp_tensor)
#
#         x_conv1 = self.conv1(x)
#         x_conv2 = self.conv2(x_conv1)
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)
#
#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(x_conv4)
#
#         batch_dict.update({
#             'encoded_spconv_tensor': out,
#             'encoded_spconv_tensor_stride': 8
#         })
#         batch_dict.update({
#             'multi_scale_3d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#             }
#         })
#
#         batch_dict.update({
#             'multi_scale_3d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#             }
#         })
#
#         return batch_dict
from functools import partial
import mindspore.nn
import torch
from mindspore import Tensor,ops
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m
class VoxelBackBone8x(mindspore.nn.Cell):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }


    def load_param(self,model_state_disk,num):

        conv_input,conv1,conv2,conv3,conv4,conv_out={},{},{},{},{},{}
        for key,value in model_state_disk.items():
            if 'pfe' in key:
                break
            if 'backbone_3d' in key:
                key=key[len('backbone_3d.'):]
                if 'conv_input' in key:
                    key = key[len('conv_input.'):]
                    conv_input[key]=value
                    num+=1
                    continue
                if 'conv1' in key:
                    key = key[len('conv1.'):]
                    conv1[key]=value
                    num+=1
                    continue
                if 'conv2' in key:
                    key = key[len('conv2.'):]
                    conv2[key]=value
                    num+=1
                    continue
                if 'conv3' in key:
                    key = key[len('conv3.'):]
                    conv3[key]=value
                    num+=1
                    continue
                if 'conv4' in key:
                    key = key[len('conv4.'):]
                    conv4[key]=value
                    num+=1
                    continue
                if 'conv_out' in key:
                    key = key[len('conv_out.'):]
                    conv_out[key]=value
                    num+=1
                    continue
        self.conv_input.load_state_dict(conv_input)
        self.conv1.load_state_dict(conv1)
        self.conv2.load_state_dict(conv2)
        self.conv3.load_state_dict(conv3)
        self.conv4.load_state_dict(conv4)
        self.conv_out.load_state_dict(conv_out)
    def construct(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        voxel_features=torch.tensor(voxel_features.asnumpy())
        voxel_coords=torch.tensor(voxel_coords.asnumpy())

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']#是sptensor
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = Tensor(spatial_features.detach().numpy() )
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # x_conv1=Tensor(x_conv1.detach().numpy())
        # x_conv2=Tensor(x_conv2.detach().numpy())
        # x_conv3=Tensor(x_conv3.detach().numpy())
        # x_conv4=Tensor(x_conv4.detach().numpy())
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict