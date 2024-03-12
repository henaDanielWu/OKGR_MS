import numpy as np

from . import box_utils
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class SigmoidFocalClassificationLoss(nn.Cell):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = x2ms_adapter.clamp(input, min=0) - input * target + \
               x2ms_adapter.log1p(x2ms_adapter.exp(-x2ms_adapter.x2ms_abs(input)))
        return loss

    def construct(self, input, target, weights):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = x2ms_adapter.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * x2ms_adapter.x2ms_pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = x2ms_adapter.tensor_api.unsqueeze(weights, -1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Cell):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = x2ms_adapter.from_numpy(self.code_weights)

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = x2ms_adapter.x2ms_abs(diff)
        else:
            n = x2ms_adapter.x2ms_abs(diff)
            loss = x2ms_adapter.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def construct(self, input, target, weights = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = x2ms_adapter.where(x2ms_adapter.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * x2ms_adapter.tensor_api.view(self.code_weights, 1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * x2ms_adapter.tensor_api.unsqueeze(weights, -1)

        return loss


class WeightedL1Loss(nn.Cell):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = x2ms_adapter.from_numpy(self.code_weights)

    def construct(self, input, target, weights = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = x2ms_adapter.where(x2ms_adapter.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * x2ms_adapter.tensor_api.view(self.code_weights, 1, 1, -1)

        loss = x2ms_adapter.x2ms_abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * x2ms_adapter.tensor_api.unsqueeze(weights, -1)

        return loss


class WeightedCrossEntropyLoss(nn.Cell):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def construct(self, input, target, weights):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = x2ms_adapter.tensor_api.permute(input, 0, 2, 1)
        target = x2ms_adapter.tensor_api.argmax(target, dim=-1)
        loss = x2ms_adapter.nn_functional.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d, gt_bbox3d):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = x2ms_adapter.tensor_api.clone(gt_bbox3d)
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = x2ms_adapter.x2ms_min(x2ms_adapter.norm(pred_box_corners - gt_box_corners, dim=2),
                            x2ms_adapter.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return x2ms_adapter.tensor_api.x2ms_mean(corner_loss, dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=x2ms_adapter.Device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = x2ms_adapter.zeros(shape, dtype=mindspore.bool_, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = x2ms_adapter.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = x2ms_adapter.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = x2ms_adapter.tensor_api.long(gt_boxes2d)

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.eq(gt, 1))
    neg_inds = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.lt(gt, 1))

    neg_weights = x2ms_adapter.x2ms_pow(1 - gt, 4)

    loss = 0

    pos_loss = x2ms_adapter.log(pred) * x2ms_adapter.x2ms_pow(1 - pred, 2) * pos_inds
    neg_loss = x2ms_adapter.log(1 - pred) * x2ms_adapter.x2ms_pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = x2ms_adapter.tensor_api.x2ms_float(mask[:, None, :, :])
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = x2ms_adapter.tensor_api.x2ms_sum((x2ms_adapter.tensor_api.x2ms_float(pos_inds) * mask))
    else:
        num_pos = x2ms_adapter.tensor_api.x2ms_sum(x2ms_adapter.tensor_api.x2ms_float(pos_inds))

    pos_loss = x2ms_adapter.tensor_api.x2ms_sum(pos_loss)
    neg_loss = x2ms_adapter.tensor_api.x2ms_sum(neg_loss)

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Cell):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def construct(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = x2ms_adapter.tensor_api.x2ms_sum(x2ms_adapter.tensor_api.x2ms_float(mask))
    mask = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.expand_as(x2ms_adapter.tensor_api.unsqueeze(mask, 2), gt_regr))
    isnotnan = x2ms_adapter.tensor_api.x2ms_float((~ x2ms_adapter.isnan(gt_regr)))
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = x2ms_adapter.x2ms_abs(regr - gt_regr)
    loss = x2ms_adapter.tensor_api.transpose(loss, 2, 0)

    loss = x2ms_adapter.x2ms_sum(loss, dim=2)
    loss = x2ms_adapter.x2ms_sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    # loss = loss / torch.clamp_min(num, min=1.0)
    loss = loss / mindspore.ops.clamp(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = x2ms_adapter.tensor_api.x2ms_size(feat, 2)
    ind  = x2ms_adapter.tensor_api.expand(x2ms_adapter.tensor_api.unsqueeze(ind, 2), x2ms_adapter.tensor_api.x2ms_size(ind, 0), x2ms_adapter.tensor_api.x2ms_size(ind, 1), dim)
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


class RegLossCenterNet(nn.Cell):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def construct(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss