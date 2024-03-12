import numpy as np

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class RoIHeadTemplate(nn.Cell):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

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
                x2ms_nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                x2ms_nn.BatchNorm1d(fc_list[k]),
                x2ms_nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(x2ms_nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(x2ms_nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = x2ms_nn.Sequential(*fc_layers)
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
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = x2ms_adapter.tensor_api.new_zeros(batch_box_preds, (batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = x2ms_adapter.tensor_api.new_zeros(batch_box_preds, (batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = x2ms_adapter.tensor_api.new_zeros(batch_box_preds, (batch_size, nms_config.NMS_POST_MAXSIZE), dtype=mindspore.int64)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = x2ms_adapter.x2ms_max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        targets_dict = x2ms_adapter.forward(self.proposal_target_layer, batch_dict)

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
