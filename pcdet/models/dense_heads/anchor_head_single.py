import mindspore.ops as ops
import numpy as np
import mindspore
import mindspore.nn as nn
import x2ms_adapter
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator

class AnchorHeadSingle(nn.Cell):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
        )
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1,
            has_bias=True
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1,
            has_bias=True
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
                has_bias=True
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        x2ms_adapter.nn_init.constant_(self.conv_cls.bias, -x2ms_adapter.tensor_api.log(np, (1 - pi) / pi))
        x2ms_adapter.nn_init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def construct(self, data_dict):

        spatial_features_2d = data_dict.get('spatial_features_2d')

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1)# [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1)  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1)
            self.forward_ret_dict.update({'dir_cls_preds': dir_cls_preds})
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict.get('batch_size'),
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            # data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict.update({'batch_cls_preds': batch_cls_preds})
            data_dict.update({'batch_box_preds':batch_box_preds})
            data_dict.update({'cls_preds_normalized':False})
            # data_dict['cls_preds_normalized'] = False

        return data_dict



    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = x2ms_adapter.tensor_api.new_zeros(anchors, [*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = x2ms_adapter.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = x2ms_adapter.tensor_api.x2ms_float((negative_cls_weights + 1.0 * positives))
        reg_weights = x2ms_adapter.tensor_api.x2ms_float(positives)
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = x2ms_adapter.tensor_api.x2ms_float(
            x2ms_adapter.tensor_api.x2ms_sum(positives, 1, keepdim=True))
        reg_weights /= x2ms_adapter.clamp(pos_normalizer, min=1.0)
        cls_weights /= x2ms_adapter.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * x2ms_adapter.tensor_api.type_as(cared, box_cls_labels)
        cls_targets = x2ms_adapter.tensor_api.unsqueeze(cls_targets, dim=-1)

        cls_targets = x2ms_adapter.tensor_api.squeeze(cls_targets, dim=-1)
        one_hot_targets = x2ms_adapter.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype
        )
        x2ms_adapter.tensor_api.scatter_(one_hot_targets, -1, x2ms_adapter.tensor_api.long(
            x2ms_adapter.tensor_api.unsqueeze(cls_targets, dim=-1)), 1.0)
        cls_preds = x2ms_adapter.tensor_api.view(cls_preds, batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = x2ms_adapter.tensor_api.x2ms_sum(cls_loss_src) / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': x2ms_adapter.tensor_api.item(cls_loss)
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = x2ms_adapter.sin(boxes1[..., dim:dim + 1]) * x2ms_adapter.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = x2ms_adapter.cos(boxes1[..., dim:dim + 1]) * x2ms_adapter.sin(boxes2[..., dim:dim + 1])
        boxes1 = x2ms_adapter.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = x2ms_adapter.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = x2ms_adapter.tensor_api.view(anchors, batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = x2ms_adapter.tensor_api.long(x2ms_adapter.floor(offset_rot / (2 * np.pi / num_bins)))
        dir_cls_targets = x2ms_adapter.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = x2ms_adapter.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype)
            x2ms_adapter.tensor_api.scatter_(dir_targets, -1, x2ms_adapter.tensor_api.long(
                x2ms_adapter.tensor_api.unsqueeze(dir_cls_targets, dim=-1)), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = x2ms_adapter.tensor_api.x2ms_float(positives)
        pos_normalizer = x2ms_adapter.tensor_api.x2ms_float(
            x2ms_adapter.tensor_api.x2ms_sum(positives, 1, keepdim=True))
        reg_weights /= x2ms_adapter.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = x2ms_adapter.cat(
                    [x2ms_adapter.tensor_api.view(
                        x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(anchor, 3, 4, 0, 1, 2, 5)),
                        -1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = x2ms_adapter.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = x2ms_adapter.tensor_api.repeat(x2ms_adapter.tensor_api.view(anchors, 1, -1, anchors.shape[-1]),
                                                 batch_size, 1, 1)
        box_preds = x2ms_adapter.tensor_api.view(box_preds, batch_size, -1,
                                                 box_preds.shape[
                                                     -1] // self.num_anchors_per_location if not self.use_multihead else
                                                 box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = x2ms_adapter.tensor_api.x2ms_sum(loc_loss_src) / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': x2ms_adapter.tensor_api.item(loc_loss)
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = x2ms_adapter.tensor_api.view(box_dir_cls_preds, batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = x2ms_adapter.tensor_api.type_as(positives, dir_logits)
            weights /= x2ms_adapter.clamp(x2ms_adapter.tensor_api.x2ms_sum(weights, -1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = x2ms_adapter.tensor_api.x2ms_sum(dir_loss) / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = x2ms_adapter.tensor_api.item(dir_loss)

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = x2ms_adapter.tensor_api.item(rpn_loss)
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = x2ms_adapter.cat([x2ms_adapter.tensor_api.view(
                    x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(anchor, 3, 4, 0, 1, 2, 5)), -1,
                    anchor.shape[-1])
                                            for anchor in self.anchors], dim=0)
            else:
                anchors = x2ms_adapter.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = x2ms_adapter.tensor_api.view(anchors, -1, anchors.shape[-1]).shape[0]
        batch_anchors = x2ms_adapter.tensor_api.repeat(x2ms_adapter.tensor_api.view(anchors, 1, -1, anchors.shape[-1]),
                                                       batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else mindspore.ops.cat(box_preds, axis=1).view(batch_size, num_anchors, -1)

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.get("DIR_OFFSET", 0.78539)
            dir_limit_offset = self.model_cfg.get('DIR_LIMIT_OFFSET', 0.0)
            dir_cls_preds = x2ms_adapter.tensor_api.view(dir_cls_preds, batch_size, num_anchors, -1) if not isinstance(
                dir_cls_preds, list) \
                else x2ms_adapter.tensor_api.view(x2ms_adapter.cat(dir_cls_preds, dim=1), batch_size, num_anchors, -1)
            dir_labels = mindspore.ops.max(dir_cls_preds, axis=-1)[1]

            period = (2 * np.pi / self.model_cfg.get('NUM_DIR_BINS', 2))

            
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

