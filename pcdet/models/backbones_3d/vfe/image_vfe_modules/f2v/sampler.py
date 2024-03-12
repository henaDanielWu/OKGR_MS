from functools import partial
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class Sampler(nn.Cell):

    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode: string, Sampling mode [bilinear/nearest]
            padding_mode: string, Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

        if mindspore.__version__ >= '1.3':
            self.grid_sample = partial(x2ms_adapter.nn_functional.grid_sample, align_corners=True)
        else:
            self.grid_sample = x2ms_adapter.nn_functional.grid_sample

    def construct(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features: (B, C, D, H, W), Input frustum features
            grid: (B, X, Y, Z, 3), Sampling grids for input features
        Returns
            output_features: (B, C, X, Y, Z) Output voxel features
        """
        # Sample from grid
        output = self.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output
