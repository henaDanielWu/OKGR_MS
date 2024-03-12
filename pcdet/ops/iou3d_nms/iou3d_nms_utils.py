"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import numpy as np
import torch
from mindspore import ops
from ...utils import common_utils
from . import iou3d_nms_cuda
import mindspore
import x2ms_adapter

def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    print('iou3d.boxes_bev_iou_cpu')
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    # assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = x2ms_adapter.tensor_api.new_zeros(boxes_a, x2ms_adapter.Size((boxes_a.shape[0], boxes_b.shape[0])))
    ans_iou=torch.Tensor(x2ms_adapter.tensor_api.numpy(ans_iou))
    boxes_a=torch.Tensor(x2ms_adapter.tensor_api.numpy(boxes_a))
    boxes_b=torch.Tensor(x2ms_adapter.tensor_api.numpy(boxes_b))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a, boxes_b, ans_iou)
    ans_iou=ans_iou.numpy()
    ans_iou=mindspore.Tensor(ans_iou)
    # 还是要求输入torch的tensor
    return x2ms_adapter.tensor_api.numpy(ans_iou) if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    print('iou3d.boxes_iou_bev')
    boxes_a.shape[1] ==7 and boxes_b.shape[1] == 7
    ans_iou = x2ms_adapter.tensor_api.zero_(x2ms_adapter.FloatTensor(x2ms_adapter.Size((boxes_a.shape[0], boxes_b.shape[0]))))
    boxes_a=torch.Tensor(boxes_a.asnumpy())
    boxes_b=torch.Tensor(boxes_b.asnumpy())
    ans_iou=torch.Tensor(ans_iou.asnumpy())
    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou)
    ans_iou=ans_iou.numpy()
    ans_iou=mindspore.Tensor(ans_iou)
    return ans_iou



def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    # print('iou3d.boxes_iou3d_gpu')
    assert boxes_a.shape[1] ==7 and boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # # bev overlap
    overlaps_bev = torch.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()
    boxes_a=torch.Tensor(boxes_a.asnumpy())
    boxes_b=torch.Tensor(boxes_b.asnumpy())
    over=iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.cuda(), boxes_b.cuda(), overlaps_bev.cuda())
    over=mindspore.tensor(over.cpu().numpy())
    boxes_a=mindspore.Tensor(boxes_a.numpy(),mindspore.float32)
    boxes_b=mindspore.Tensor(boxes_b.numpy(),mindspore.float32)

    max_of_min = ops.maximum(boxes_a_height_min, boxes_b_height_min)
    min_of_max = ops.minimum(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = x2ms_adapter.clamp(min_of_max - max_of_min, 0)

    # # 3d iou
    overlaps_3d = over * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / x2ms_adapter.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # print('iou3d.nms_gpu')
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order]
    keep = torch.LongTensor(boxes.shape[0])
    boxes=torch.tensor(boxes.asnumpy())
    out = iou3d_nms_cuda.nms_gpu(boxes.cuda(), keep, thresh)
    num_out,keep=out[:2]
    keep=mindspore.tensor(keep.cpu().numpy())
    return order[keep[:num_out.item()]], None
#     keep = ops.zeros((boxes.shape[0]),dtype=mindspore.int64)
#     boxes = torch.tensor(x2ms_adapter.tensor_api.numpy(boxes))
#     keep=torch.tensor(x2ms_adapter.tensor_api.numpy(keep))
#     # thresh=torch.Tensor((thresh,))
# #     num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    # def nms_nd_pytorch(dets: torch.Tensor, threshold: float):
    #     """
    #     :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
    #                    [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    #     :param threshold: for example 0.5
    #     :return: the rest ids of dets
    #     """
    #     dim = dets.shape[-1] // 2
    #     assert dim in (2, 3), dets.shape

    #     scores = dets[:, -1].clone()
    #     bboxes = dets[:, :-1].clone()
    #     assert bboxes.shape[-1] == 2 * dim, bboxes.shape

    #     area = torch.prod(bboxes[:, dim:] - bboxes[:, :dim] + 1, dim=-1).float()
    #     # print(area)

    #     order = scores.argsort(descending=True)

    #     keep = []
    #     while order.shape[0] > 0:
    #         i = order[0]
    #         keep.append(i.item())

    #         overlap = torch.min(bboxes[i, dim:], bboxes[order[1:]][:, dim:])
    #         overlap = overlap - torch.max(bboxes[i, :dim], bboxes[order[1:]][:, :dim]) + 1
    #         overlap = torch.clamp(overlap, min=0)
    #         inter = torch.prod(overlap, dim=-1).float()
    #         # print(inters)

    #         union = area[i] + area[order[1:]] - inter
    #         iou = inter / union
    #         # print(iou)

    #         index = torch.where(iou <= threshold)[0]
    #         # print(index)

    #         # similar to soft nms_nd
    #         # weight = torch.exp(-(iou * iou) / 0.5)
    #         # scores[order[1:]] = weight * scores[order[1:]]

    #         order = order[index + 1]

    #     dets = torch.cat((bboxes, scores.unsqueeze(-1)), dim=1)
    #     keep = torch.tensor(keep)
    #     return keep, dets
    # keep,_=nms_nd_pytorch(boxes,thresh)
    # keep=mindspore.Tensor(keep.detach().numpy())
    # return order[keep.tolist()], None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    print('iou3d.nms_normal_gpu')
    assert boxes.shape[1] == 7
    order = x2ms_adapter.tensor_api.sort(scores, 0, descending=True)[1]

    boxes = x2ms_adapter.tensor_api.contiguous(boxes[order])

    keep = x2ms_adapter.LongTensor(x2ms_adapter.tensor_api.x2ms_size(boxes, 0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return x2ms_adapter.tensor_api.contiguous(order[keep[:num_out]]), None
