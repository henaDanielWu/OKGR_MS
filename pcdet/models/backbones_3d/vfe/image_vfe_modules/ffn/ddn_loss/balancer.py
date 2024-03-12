
from pcdet.utils import loss_utils
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class Balancer(nn.Cell):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def construct(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss: (B, H, W), Pixel-wise loss
            gt_boxes2d: (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                             shape=loss.shape,
                                             downsample_factor=self.downsample_factor,
                                             device=loss.device)
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = x2ms_adapter.tensor_api.x2ms_sum(fg_mask) + x2ms_adapter.tensor_api.x2ms_sum(bg_mask)

        # Compute losses
        loss *= weights
        fg_loss = x2ms_adapter.tensor_api.x2ms_sum(loss[fg_mask]) / num_pixels
        bg_loss = x2ms_adapter.tensor_api.x2ms_sum(loss[bg_mask]) / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        tb_dict = {"balancer_loss": x2ms_adapter.tensor_api.item(loss), "fg_loss": x2ms_adapter.tensor_api.item(fg_loss), "bg_loss": x2ms_adapter.tensor_api.item(bg_loss)}
        return loss, tb_dict
