import numpy as np
from mindspore import ops,Tensor
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
import mindspore.nn as nn
import x2ms_adapter
from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
import mindspore

class PVRCNNHead(nn.Cell):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, has_bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(p=self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.SequentialCell(*shared_fc_list)

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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
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

        targets_dict = self.proposal_layer(#里面都是列表
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

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, has_bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(p=self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, has_bias=True))
        fc_layers = nn.SequentialCell(*fc_layers)
        return fc_layers

    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict

        batch_size = batch_dict.get('batch_size')
        batch_box_preds = batch_dict.get('batch_box_preds')
        batch_cls_preds = batch_dict.get('batch_cls_preds')
        # a=nms_config.NMS_POST_MAXSIZE
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        # rois, roi_scores, roi_labels=[],[],[]
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict.get('batch_index') == index)
            else:
                assert batch_dict.get('batch_cls_preds').shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = mindspore.ops.max(cls_preds, axis=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )
            # self.rois = Tensor(np.zeros([batch_size, a, batch_box_preds.shape[-1]]).astype(np.int32))
            # roi_scores = np.zeros((batch_size, a))
            # roi_labels = np.zeros((batch_size, a))
            # self.rois.append(box_preds[selected])
            # roi_scores[index, :len(selected)] = cur_roi_scores[selected].asnumpy()
            # roi_labels[index, :len(selected)] = cur_roi_labels[selected].asnumpy()
            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]



        # batch_dict['rois'] =rois
        # batch_dict['roi_scores'] = roi_scores
        # batch_dict['roi_labels'] = roi_labels
        # batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.update({
            'rois':rois,
            'roi_scores':roi_scores,
            'roi_labels':roi_labels,
            'has_class_labels':True if batch_cls_preds.shape[-1] > 1 else False
        })
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict.get('batch_size')
        targets_dict = self.proposal_target_layer(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = x2ms_adapter.tensor_api.detach(x2ms_adapter.tensor_api.clone(gt_of_rois))

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = x2ms_adapter.tensor_api.view(common_utils.rotate_points_along_z(
            points=x2ms_adapter.tensor_api.view(gt_of_rois, -1, 1, gt_of_rois.shape[-1]), angle=-x2ms_adapter.tensor_api.view(roi_ry, -1)
        ), batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = x2ms_adapter.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = x2ms_adapter.tensor_api.view(forward_ret_dict['reg_valid_mask'], -1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = x2ms_adapter.tensor_api.view(forward_ret_dict['gt_of_rois_src'][..., 0:code_size], -1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = x2ms_adapter.tensor_api.view(gt_boxes3d_ct, -1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.x2ms_sum(x2ms_adapter.tensor_api.long(fg_mask)))

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.detach(x2ms_adapter.tensor_api.clone(roi_boxes3d)), -1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                x2ms_adapter.tensor_api.view(gt_boxes3d_ct, rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.view(rcnn_reg, rcnn_batch_size, -1), dim=0),
                x2ms_adapter.tensor_api.unsqueeze(reg_targets, dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = x2ms_adapter.tensor_api.x2ms_sum((x2ms_adapter.tensor_api.view(rcnn_loss_reg, rcnn_batch_size, -1) * x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.unsqueeze(fg_mask, dim=-1)))) / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = x2ms_adapter.tensor_api.item(rcnn_loss_reg)

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = x2ms_adapter.tensor_api.view(rcnn_reg, rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = x2ms_adapter.tensor_api.view(roi_boxes3d, -1, code_size)[fg_mask]

                fg_roi_boxes3d = x2ms_adapter.tensor_api.view(fg_roi_boxes3d, 1, -1, code_size)
                batch_anchors = x2ms_adapter.tensor_api.detach(x2ms_adapter.tensor_api.clone(fg_roi_boxes3d))
                roi_ry = x2ms_adapter.tensor_api.view(fg_roi_boxes3d[:, :, 6], -1)
                roi_xyz = x2ms_adapter.tensor_api.view(fg_roi_boxes3d[:, :, 0:3], -1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = x2ms_adapter.tensor_api.view(self.box_coder.decode_torch(
                    x2ms_adapter.tensor_api.view(fg_rcnn_reg, batch_anchors.shape[0], -1, code_size), batch_anchors
                ), -1, code_size)

                rcnn_boxes3d = x2ms_adapter.tensor_api.squeeze(common_utils.rotate_points_along_z(
                    x2ms_adapter.tensor_api.unsqueeze(rcnn_boxes3d, dim=1), roi_ry
                ), dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = x2ms_adapter.tensor_api.x2ms_mean(loss_corner)
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = x2ms_adapter.tensor_api.item(loss_corner)
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = x2ms_adapter.tensor_api.view(forward_ret_dict['rcnn_cls_labels'], -1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = x2ms_adapter.tensor_api.view(rcnn_cls, -1)
            batch_loss_cls = x2ms_adapter.nn_functional.binary_cross_entropy(x2ms_adapter.sigmoid(x2ms_adapter.tensor_api.x2ms_float(rcnn_cls_flat)), x2ms_adapter.tensor_api.x2ms_float(rcnn_cls_labels), reduction='none')
            cls_valid_mask = x2ms_adapter.tensor_api.x2ms_float((rcnn_cls_labels >= 0))
            rcnn_loss_cls = x2ms_adapter.tensor_api.x2ms_sum((batch_loss_cls * cls_valid_mask)) / x2ms_adapter.clamp(x2ms_adapter.tensor_api.x2ms_sum(cls_valid_mask), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = x2ms_adapter.nn_functional.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = x2ms_adapter.tensor_api.x2ms_float((rcnn_cls_labels >= 0))
            rcnn_loss_cls = x2ms_adapter.tensor_api.x2ms_sum((batch_loss_cls * cls_valid_mask)) / x2ms_adapter.clamp(x2ms_adapter.tensor_api.x2ms_sum(cls_valid_mask), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': x2ms_adapter.tensor_api.item(rcnn_loss_cls)}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = x2ms_adapter.tensor_api.item(rcnn_loss)
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = x2ms_adapter.tensor_api.view(cls_preds, batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = x2ms_adapter.tensor_api.view(box_preds, batch_size, -1, code_size)

        roi_ry = x2ms_adapter.tensor_api.view(rois[:, :, 6], -1)
        roi_xyz = x2ms_adapter.tensor_api.view(rois[:, :, 0:3], -1, 3)
        local_rois = x2ms_adapter.tensor_api.detach(x2ms_adapter.tensor_api.clone(rois))
        local_rois[:, :, 0:3] = 0

        batch_box_preds = x2ms_adapter.tensor_api.view(self.box_coder.decode_torch(batch_box_preds, local_rois), -1, code_size)

        batch_box_preds = x2ms_adapter.tensor_api.squeeze(common_utils.rotate_points_along_z(
            x2ms_adapter.tensor_api.unsqueeze(batch_box_preds, dim=1), roi_ry
        ), dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = x2ms_adapter.tensor_api.view(batch_box_preds, batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
