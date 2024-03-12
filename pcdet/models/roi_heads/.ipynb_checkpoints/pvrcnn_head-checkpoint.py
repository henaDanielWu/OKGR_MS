
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import mindspore
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                x2ms_nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                x2ms_nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                x2ms_nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(x2ms_nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = x2ms_nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = x2ms_adapter.nn_init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = x2ms_adapter.nn_init.xavier_normal_
        elif weight_init == 'normal':
            init_func = x2ms_adapter.nn_init.normal_
        else:
            raise NotImplementedError

        for m in x2ms_adapter.nn_cell.modules(self):
            if isinstance(m, x2ms_nn.Conv2d) or isinstance(m, x2ms_nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    x2ms_adapter.nn_init.constant_(m.bias, 0)
        x2ms_adapter.nn_init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * x2ms_adapter.tensor_api.view(batch_dict['point_cls_scores'], -1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = x2ms_adapter.tensor_api.view(global_roi_grid_points, batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = x2ms_adapter.tensor_api.x2ms_int(x2ms_adapter.tensor_api.new_zeros(xyz, batch_size))
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = x2ms_adapter.tensor_api.x2ms_sum((batch_idx == k))

        new_xyz = x2ms_adapter.tensor_api.view(global_roi_grid_points, -1, 3)
        new_xyz_batch_cnt = x2ms_adapter.tensor_api.fill_(x2ms_adapter.tensor_api.x2ms_int(x2ms_adapter.tensor_api.new_zeros(xyz, batch_size)), global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=x2ms_adapter.tensor_api.contiguous(xyz),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=x2ms_adapter.tensor_api.contiguous(point_features),
            weights=None,
        )  # (M1 + M2 ..., C)

        pooled_features = x2ms_adapter.tensor_api.view(
            pooled_features, -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = x2ms_adapter.tensor_api.view(rois, -1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = x2ms_adapter.tensor_api.squeeze(common_utils.rotate_points_along_z(
            x2ms_adapter.tensor_api.clone(local_roi_grid_points), rois[:, 6]
        ), dim=1)
        global_center = x2ms_adapter.tensor_api.clone(rois[:, 0:3])
        global_roi_grid_points += x2ms_adapter.tensor_api.unsqueeze(global_center, dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = x2ms_adapter.tensor_api.new_ones(rois, (grid_size, grid_size, grid_size))
        dense_idx = x2ms_adapter.tensor_api.nonzero(faked_features)  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.repeat(dense_idx, batch_size_rcnn, 1, 1))  # (B, 6x6x6, 3)

        local_roi_size = x2ms_adapter.tensor_api.view(rois, batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * x2ms_adapter.tensor_api.unsqueeze(local_roi_size, dim=1) \
                          - (x2ms_adapter.tensor_api.unsqueeze(local_roi_size, dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def construct(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(pooled_features, 0, 2, 1)), batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(x2ms_adapter.tensor_api.view(pooled_features, batch_size_rcnn, -1, 1))
        rcnn_cls = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.transpose(self.cls_layers(shared_features), 1, 2)), dim=1)  # (B, 1 or 2)
        rcnn_reg = x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.transpose(self.reg_layers(shared_features), 1, 2)), dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
