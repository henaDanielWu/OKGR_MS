import copy
import numpy as np
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class SeparateHead(nn.Cell):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(x2ms_nn.Sequential(
                    x2ms_nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    x2ms_nn.BatchNorm2d(input_channels),
                    x2ms_nn.ReLU()
                ))
            fc_list.append(x2ms_nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = x2ms_nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                x2ms_adapter.tensor_api.fill_(fc[-1].bias.data, init_bias)
            else:
                for m in x2ms_adapter.nn_cell.modules(fc):
                    if isinstance(m, x2ms_nn.Conv2d):
                        x2ms_adapter.nn_init.kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            x2ms_adapter.nn_init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def construct(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Cell):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = x2ms_adapter.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            ))
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            x2ms_nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            x2ms_nn.ReLU(),
        )

        self.heads_list = x2ms_nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = x2ms_adapter.tensor_api.new_zeros(gt_boxes, num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = x2ms_adapter.tensor_api.new_zeros(gt_boxes, (num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = x2ms_adapter.tensor_api.long(x2ms_adapter.tensor_api.new_zeros(gt_boxes, num_max_objs))
        mask = x2ms_adapter.tensor_api.long(x2ms_adapter.tensor_api.new_zeros(gt_boxes, num_max_objs))

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = x2ms_adapter.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = x2ms_adapter.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = x2ms_adapter.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = x2ms_adapter.tensor_api.x2ms_int(center)
        center_int_float = x2ms_adapter.tensor_api.x2ms_float(center_int)

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        # radius = torch.clamp_min(x2ms_adapter.tensor_api.x2ms_int(radius), min=min_radius)
        radius = mindspore.ops.clamp(x2ms_adapter.tensor_api.x2ms_int(radius), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = x2ms_adapter.tensor_api.long((gt_boxes[k, -1] - 1))
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], x2ms_adapter.tensor_api.item(radius[k]))

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - x2ms_adapter.tensor_api.x2ms_float(center_int_float[k])
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = x2ms_adapter.tensor_api.log(gt_boxes[k, 3:6])
            ret_boxes[k, 6] = x2ms_adapter.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = x2ms_adapter.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[x2ms_adapter.tensor_api.numpy(x2ms_adapter.tensor_api.long(cur_gt_boxes[:, -1]))]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = x2ms_adapter.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(x2ms_adapter.to(heatmap, gt_boxes_single_head.device))
                target_boxes_list.append(x2ms_adapter.to(ret_boxes, gt_boxes_single_head.device))
                inds_list.append(x2ms_adapter.to(inds, gt_boxes_single_head.device))
                masks_list.append(x2ms_adapter.to(mask, gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(x2ms_adapter.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(x2ms_adapter.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(x2ms_adapter.stack(inds_list, dim=0))
            ret_dict['masks'].append(x2ms_adapter.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = x2ms_adapter.clamp(x2ms_adapter.tensor_api.sigmoid(x), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = x2ms_adapter.tensor_api.sigmoid(self, pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = x2ms_adapter.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = x2ms_adapter.tensor_api.x2ms_sum((reg_loss * x2ms_adapter.tensor_api.new_tensor(reg_loss, self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])))
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = x2ms_adapter.tensor_api.item(hm_loss)
            tb_dict['loc_loss_head_%d' % idx] = x2ms_adapter.tensor_api.item(loc_loss)

        tb_dict['rpn_loss'] = x2ms_adapter.tensor_api.item(loss)
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.x2ms_tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE))

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = x2ms_adapter.tensor_api.sigmoid(pred_dict['hm'])
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = x2ms_adapter.tensor_api.exp(pred_dict['dim'])
            batch_rot_cos = x2ms_adapter.tensor_api.unsqueeze(pred_dict['rot'][:, 0], dim=1)
            batch_rot_sin = x2ms_adapter.tensor_api.unsqueeze(pred_dict['rot'][:, 1], dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][x2ms_adapter.tensor_api.long(final_dict['pred_labels'])]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = x2ms_adapter.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = x2ms_adapter.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = x2ms_adapter.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = x2ms_adapter.tensor_api.new_zeros(pred_boxes, (batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = x2ms_adapter.tensor_api.new_zeros(pred_boxes, (batch_size, num_max_rois))
        roi_labels = x2ms_adapter.tensor_api.long(x2ms_adapter.tensor_api.new_zeros(pred_boxes, (batch_size, num_max_rois)))

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def construct(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=x2ms_adapter.tensor_api.x2ms_size(spatial_features_2d)[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
