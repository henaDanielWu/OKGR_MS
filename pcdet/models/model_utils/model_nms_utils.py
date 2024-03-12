
from ...ops.iou3d_nms import iou3d_nms_utils
import mindspore
import x2ms_adapter
import torch

def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    scores_mask=None
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores=torch.Tensor(box_scores.asnumpy())
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        indices=mindspore.Tensor(indices.numpy())
        box_scores_nms=mindspore.Tensor(box_scores_nms.numpy())
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, 'nms_gpu')(
            boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
    if len(selected)==0 and score_thresh is not None:
        score,select=mindspore.ops.topk(src_box_scores,k=2)
        return select,score
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = x2ms_adapter.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(x2ms_adapter.tensor_api.long(x2ms_adapter.tensor_api.new_ones(box_scores, len(selected))) * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = x2ms_adapter.cat(pred_scores, dim=0)
    pred_labels = x2ms_adapter.cat(pred_labels, dim=0)
    pred_boxes = x2ms_adapter.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
