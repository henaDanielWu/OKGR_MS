from pcdet.models.backbones_3d.completion_models.utils import query_knn, grouping_operation
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

class Transformer(nn.Cell):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = x2ms_nn.Conv1d(dim, dim, 1)
        self.conv_query = x2ms_nn.Conv1d(dim, dim, 1)
        self.conv_value = x2ms_nn.Conv1d(dim, dim, 1)

        self.pos_mlp = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(3, pos_hidden_dim, 1),
            x2ms_nn.BatchNorm2d(pos_hidden_dim),
            x2ms_nn.ReLU(),
            x2ms_nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = x2ms_nn.Sequential(
            x2ms_nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            x2ms_nn.BatchNorm2d(dim * attn_hidden_multiplier),
            x2ms_nn.ReLU(),
            x2ms_nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = x2ms_nn.Conv1d(in_channel, dim, 1)
        self.linear_end = x2ms_nn.Conv1d(dim, in_channel, 1)

    def construct(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(pos, 0, 2, 1))
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key  # b, dim, n, n_knn

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, n, n_knn
        attention = x2ms_adapter.softmax(attention, -1)  # b, dim, n, n_knn

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = x2ms_adapter.einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity