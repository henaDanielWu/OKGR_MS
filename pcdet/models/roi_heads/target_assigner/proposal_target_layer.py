import numpy as np
import torch
from ....ops.iou3d_nms import iou3d_nms_utils
import mindspore
import mindspore.nn as nn
import x2ms_adapter
from mindspore import ops
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class ProposalTargetLayer(nn.Cell):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def construct(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # regression valid mask
        reg_valid_mask = x2ms_adapter.tensor_api.long((batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH))

        # classification label
        if self.roi_sampler_cfg.get('CLS_SCORE_TYPE','roi_iou') == 'cls':
            batch_cls_labels = x2ms_adapter.tensor_api.long((batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH))
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.get('CLS_SCORE_TYPE','roi_iou') == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = x2ms_adapter.tensor_api.x2ms_float((fg_mask > 0))
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
        
        """
        batch_size = batch_dict.get('batch_size')
        rois = batch_dict.get('rois')
        roi_scores = batch_dict.get('roi_scores')
        roi_labels = batch_dict.get('roi_labels')
        gt_boxes = batch_dict.get('gt_boxes')
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros((batch_size, self.roi_sampler_cfg.get('ROI_PER_IMAGE',128), code_size))
        batch_gt_of_rois = rois.new_zeros(( batch_size, self.roi_sampler_cfg.get('ROI_PER_IMAGE',128), code_size + 1))
        batch_roi_ious = rois.new_zeros((batch_size, self.roi_sampler_cfg.get('ROI_PER_IMAGE',128)))
        batch_roi_scores = rois.new_zeros((batch_size, self.roi_sampler_cfg.get('ROI_PER_IMAGE',128)))
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.get('ROI_PER_IMAGE',128)))

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=x2ms_adapter.tensor_api.long(cur_gt[:, -1])
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = x2ms_adapter.x2ms_max(iou3d, dim=1)

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            gt_assignment=mindspore.Tensor(gt_assignment.asnumpy(),dtype=mindspore.int32)
            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)
        temp=torch.Tensor(max_overlaps.asnumpy())
        fg_inds = ((temp >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((temp < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        hard_bg_inds = ((temp < self.roi_sampler_cfg.REG_FG_THRESH) &
                        (temp >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
        fg_inds=mindspore.Tensor(fg_inds.numpy())
        easy_bg_inds=mindspore.Tensor(easy_bg_inds.numpy())
        hard_bg_inds=mindspore.Tensor(hard_bg_inds.numpy())
        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = mindspore.Tensor(np.random.permutation(fg_num_rois),dtype=mindspore.int64)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = mindspore.Tensor(rand_num,dtype=mindspore.int64)
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = ops.cat((fg_inds, bg_inds), axis=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if x2ms_adapter.tensor_api.numel(hard_bg_inds) > 0 and x2ms_adapter.tensor_api.numel(easy_bg_inds) > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = x2ms_adapter.tensor_api.long(x2ms_adapter.randint(low=0, high=x2ms_adapter.tensor_api.numel(hard_bg_inds), size=(hard_bg_rois_num,)))
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = x2ms_adapter.tensor_api.long(x2ms_adapter.randint(low=0, high=x2ms_adapter.tensor_api.numel(easy_bg_inds), size=(easy_bg_rois_num,)))
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = x2ms_adapter.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif x2ms_adapter.tensor_api.numel(hard_bg_inds) > 0 and x2ms_adapter.tensor_api.numel(easy_bg_inds) == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = x2ms_adapter.tensor_api.long(x2ms_adapter.randint(low=0, high=x2ms_adapter.tensor_api.numel(hard_bg_inds), size=(hard_bg_rois_num,)))
            bg_inds = hard_bg_inds[rand_idx]
        elif x2ms_adapter.tensor_api.numel(hard_bg_inds) == 0 and x2ms_adapter.tensor_api.numel(easy_bg_inds) > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = x2ms_adapter.tensor_api.long(x2ms_adapter.randint(low=0, high=x2ms_adapter.tensor_api.numel(easy_bg_inds), size=(easy_bg_rois_num,)))
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:
            
        Returns:
        
        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
        # gt_labels=mindspore.Tensor(gt_labels,dtype=mindspore.float32)
        a=gt_labels.min()
        b=gt_labels.max() + 1
        # c=np.arange(a,b)
        k=a
        while k< b:
        # for k in c:
            # if not isinstance(gt_labels,mindspore.Tensor):
            #     gt_labels=mindspore.Tensor(gt_labels.numpy(),mindspore.float32)
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if x2ms_adapter.tensor_api.x2ms_sum(roi_mask) > 0 and x2ms_adapter.tensor_api.x2ms_sum(gt_mask) > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                # cur_roi = mindspore.Tensor(rois[roi_mask],mindspore.float32)
                # cur_gt = mindspore.Tensor(gt_boxes[gt_mask],mindspore.float32)
                original_gt_assignment = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.nonzero(gt_mask), -1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:, :7], cur_gt[:, :7])  # (M, N)
                cur_max_overlaps, cur_gt_assignment = x2ms_adapter.x2ms_max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]
            k+=1

        return max_overlaps, gt_assignment
