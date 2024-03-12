# This file is modified from https://github.com/tianweiy/CenterPoint

import numpy as np
import mindspore
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = x2ms_adapter.tensor_api.sqrt((b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = x2ms_adapter.tensor_api.sqrt((b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = x2ms_adapter.tensor_api.sqrt((b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    ret = x2ms_adapter.x2ms_min(x2ms_adapter.x2ms_min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = x2ms_adapter.tensor_api.exp(np, -(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * x2ms_adapter.tensor_api.x2ms_max(h)] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.to(x2ms_adapter.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ), heatmap.device))

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * x2ms_adapter.tensor_api.x2ms_float(cur_valid_mask)

        x2ms_adapter.x2ms_max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = x2ms_adapter.nn_functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = x2ms_adapter.tensor_api.x2ms_float((hmax == heat))
    return heat * keep


def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = x2ms_adapter.tensor_api.argsort(scores)[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(x2ms_adapter.tensor_api.numpy(boxes), thresh=min_radius))[:post_max_size]

    keep = x2ms_adapter.to(x2ms_adapter.tensor_api.long(x2ms_adapter.from_numpy(keep)), boxes.device)

    return keep


def _gather_feat(feat, ind, mask=None):
    dim = x2ms_adapter.tensor_api.x2ms_size(feat, 2)
    ind = x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(ind, 2), x2ms_adapter.tensor_api.x2ms_size(ind, 0), x2ms_adapter.tensor_api.x2ms_size(ind, 1), dim)
    feat = x2ms_adapter.tensor_api.gather(feat, 1, ind)
    if mask is not None:
        mask = x2ms_adapter.tensor_api.expand_as(x2ms_adapter.tensor_api.unsqueeze(mask, 2), feat)
        feat = feat[mask]
        feat = x2ms_adapter.tensor_api.view(feat, -1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(feat, 0, 2, 3, 1))
    feat = x2ms_adapter.tensor_api.view(feat, x2ms_adapter.tensor_api.x2ms_size(feat, 0), -1, x2ms_adapter.tensor_api.x2ms_size(feat, 3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, num_class, height, width = x2ms_adapter.tensor_api.x2ms_size(scores)

    topk_scores, topk_inds = x2ms_adapter.topk(x2ms_adapter.tensor_api.flatten(scores, 2, 3), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = x2ms_adapter.tensor_api.x2ms_float((topk_inds // width))
    topk_xs = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.x2ms_int((topk_inds % width)))

    topk_score, topk_ind = x2ms_adapter.topk(x2ms_adapter.tensor_api.view(topk_scores, batch, -1), K)
    topk_classes = x2ms_adapter.tensor_api.x2ms_int((topk_ind // K))
    topk_inds = x2ms_adapter.tensor_api.view(_gather_feat(x2ms_adapter.tensor_api.view(topk_inds, batch, -1, 1), topk_ind), batch, K)
    topk_ys = x2ms_adapter.tensor_api.view(_gather_feat(x2ms_adapter.tensor_api.view(topk_ys, batch, -1, 1), topk_ind), batch, K)
    topk_xs = x2ms_adapter.tensor_api.view(_gather_feat(x2ms_adapter.tensor_api.view(topk_xs, batch, -1, 1), topk_ind), batch, K)

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None):
    batch_size, num_class, _, _ = x2ms_adapter.tensor_api.x2ms_size(heatmap)

    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)

    scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)
    center = x2ms_adapter.tensor_api.view(_transpose_and_gather_feat(center, inds), batch_size, K, 2)
    rot_sin = x2ms_adapter.tensor_api.view(_transpose_and_gather_feat(rot_sin, inds), batch_size, K, 1)
    rot_cos = x2ms_adapter.tensor_api.view(_transpose_and_gather_feat(rot_cos, inds), batch_size, K, 1)
    center_z = x2ms_adapter.tensor_api.view(_transpose_and_gather_feat(center_z, inds), batch_size, K, 1)
    dim = x2ms_adapter.tensor_api.view(_transpose_and_gather_feat(dim, inds), batch_size, K, 3)

    angle = x2ms_adapter.atan2(rot_sin, rot_cos)
    xs = x2ms_adapter.tensor_api.view(xs, batch_size, K, 1) + center[:, :, 0:1]
    ys = x2ms_adapter.tensor_api.view(ys, batch_size, K, 1) + center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = x2ms_adapter.tensor_api.view(_transpose_and_gather_feat(vel, inds), batch_size, K, 2)
        box_part_list.append(vel)

    final_box_preds = x2ms_adapter.cat((box_part_list), dim=-1)
    final_scores = x2ms_adapter.tensor_api.view(scores, batch_size, K)
    final_class_ids = x2ms_adapter.tensor_api.view(class_ids, batch_size, K)

    assert post_center_limit_range is not None
    mask = x2ms_adapter.tensor_api.x2ms_all((final_box_preds[..., :3] >= post_center_limit_range[:3]), 2)
    mask &= x2ms_adapter.tensor_api.x2ms_all((final_box_preds[..., :3] <= post_center_limit_range[3:]), 2)

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]

        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = x2ms_adapter.cat((centers, x2ms_adapter.tensor_api.view(scores, -1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })
    return ret_pred_dicts
