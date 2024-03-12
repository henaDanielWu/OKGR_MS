from os import getgrouplist
import numpy as np
from typing import Optional, List
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class PointNetfeat(nn.Cell):
    def __init__(self, input_dim, x=1,outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel==256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = x2ms_nn.Conv1d(input_dim, 64 * x, 1)
        self.conv2 = x2ms_nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = x2ms_nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = x2ms_nn.Conv1d(256 * x,  self.output_channel, 1)
        self.bn1 = x2ms_nn.BatchNorm1d(64 * x)
        self.bn2 = x2ms_nn.BatchNorm1d(128 * x)
        self.bn3 = x2ms_nn.BatchNorm1d(256 * x)
        self.bn4 = x2ms_nn.BatchNorm1d(self.output_channel)

    def construct(self, x):
        x = x2ms_adapter.nn_functional.relu(self.bn1(self.conv1(x)))
        x = x2ms_adapter.nn_functional.relu(self.bn2(self.conv2(x)))
        x = x2ms_adapter.nn_functional.relu(self.bn3(self.conv3(x)))
        x_ori = self.bn4(self.conv4(x)) 

        x = x2ms_adapter.x2ms_max(x_ori, 2, keepdim=True)[0]

        x = x2ms_adapter.tensor_api.view(x, -1, self.output_channel)
        return x, x_ori

class PointNet(nn.Cell):
    def __init__(self, input_dim, joint_feat=False,model_cfg=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
        channels = model_cfg.TRANS_INPUT

        times=1
        self.feat = PointNetfeat(input_dim, 1)

        self.fc1 = x2ms_nn.Linear(512, 256 )
        self.fc2 = x2ms_nn.Linear(256, channels)

        self.pre_bn = x2ms_nn.BatchNorm1d(input_dim)
        self.bn1 = x2ms_nn.BatchNorm1d(256)
        self.bn2 = x2ms_nn.BatchNorm1d(channels)
        self.relu = x2ms_nn.ReLU()

        self.fc_s1 = x2ms_nn.Linear(channels*times, 256)
        self.fc_s2 = x2ms_nn.Linear(256, 3, bias=False)
        self.fc_ce1 = x2ms_nn.Linear(channels*times, 256)
        self.fc_ce2 = x2ms_nn.Linear(256, 3, bias=False)
        self.fc_hr1 = x2ms_nn.Linear(channels*times, 256)
        self.fc_hr2 = x2ms_nn.Linear(256, 1, bias=False)

    def construct(self, x, feat=None):

        if self.joint_feat:
            if len(feat.shape) > 2:
                feat = x2ms_adapter.x2ms_max(feat, 2, keepdim=True)[0]
                x = x2ms_adapter.tensor_api.view(feat, -1, self.output_channel)
                x = x2ms_adapter.nn_functional.relu(self.bn1(self.fc1(x)))
                feat = x2ms_adapter.nn_functional.relu(self.bn2(self.fc2(x)))
            else:
                feat = feat
            feat_traj = None
        else:
            x, feat_traj = self.feat(self.pre_bn(x))
            x = x2ms_adapter.nn_functional.relu(self.bn1(self.fc1(x)))
            feat = x2ms_adapter.nn_functional.relu(self.bn2(self.fc2(x)))

        x = x2ms_adapter.nn_functional.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = x2ms_adapter.nn_functional.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = x2ms_adapter.nn_functional.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        return x2ms_adapter.cat([centers, sizes, headings],-1),feat,feat_traj

    def init_weights(self):
        for m in x2ms_adapter.nn_cell.modules(self):
            if isinstance(m, x2ms_nn.Conv1d) or isinstance(m, x2ms_nn.Linear):
                x2ms_adapter.nn_init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    x2ms_adapter.nn_init.zeros_(m.bias)

class MLP(nn.Cell):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = x2ms_nn.ModuleList(x2ms_nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = x2ms_adapter.nn_functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SpatialMixerBlock(nn.Cell):

    def __init__(self,hidden_dim,grid_size,channels,config=None,dropout=0.0):
        super().__init__()


        self.mixer_x = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.mixer_y = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.mixer_z = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.norm_x = x2ms_nn.LayerNorm(channels)
        self.norm_y = x2ms_nn.LayerNorm(channels)
        self.norm_z = x2ms_nn.LayerNorm(channels)
        self.norm_channel = x2ms_nn.LayerNorm(channels)
        self.ffn = x2ms_nn.Sequential(
                               x2ms_nn.Linear(channels, 2*channels),
                               x2ms_nn.ReLU(),
                               x2ms_nn.Dropout(dropout),
                               x2ms_nn.Linear(2*channels, channels),
                               )
        self.config = config
        self.grid_size = grid_size

    def construct(self, src):

        src_3d = x2ms_adapter.tensor_api.view(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(src, 1,2,0)), src.shape[1],src.shape[2],
                                   self.grid_size,self.grid_size,self.grid_size)
        src_3d = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(src_3d, 0,1,4,3,2)) 
        mixed_x = self.mixer_x(src_3d)
        mixed_x = src_3d + mixed_x
        mixed_x = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(self.norm_x(x2ms_adapter.tensor_api.permute(mixed_x, 0,2,3,4,1)), 0,4,1,2,3))

        mixed_y = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(self.mixer_y(x2ms_adapter.tensor_api.permute(mixed_x, 0,1,2,4,3)), 0,1,2,4,3))
        mixed_y =  mixed_x + mixed_y
        mixed_y = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(self.norm_y(x2ms_adapter.tensor_api.permute(mixed_y, 0,2,3,4,1)), 0,4,1,2,3))

        mixed_z = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(self.mixer_z(x2ms_adapter.tensor_api.permute(mixed_y, 0,1,4,3,2)), 0,1,4,3,2))

        mixed_z =  mixed_y + mixed_z
        mixed_z = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.permute(self.norm_z(x2ms_adapter.tensor_api.permute(mixed_z, 0,2,3,4,1)), 0,4,1,2,3))

        src_mixer = x2ms_adapter.tensor_api.permute(x2ms_adapter.tensor_api.view(mixed_z, src.shape[1],src.shape[2],-1), 2,0,1)
        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)

        return src_mixer

class Transformer(nn.Cell):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,
                num_lidar_points=None,num_proxy_points=None, share_head=True,num_groups=None,
                sequence_stride=None,num_frames=None):
        super().__init__()

        self.config = config
        self.share_head = share_head
        self.num_frames = num_frames
        self.nhead = nhead
        self.sequence_stride = sequence_stride
        self.num_groups = num_groups
        self.num_proxy_points = num_proxy_points
        self.num_lidar_points = num_lidar_points
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model, nhead, dim_feedforward,dropout, activation, 
                      normalize_before, num_lidar_points,num_groups=num_groups) for i in range(num_encoder_layers)]

        encoder_norm = x2ms_nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)

        self.token = mindspore.Parameter(x2ms_adapter.zeros(self.num_groups, 1, d_model))

        
        if self.num_frames >4:
  
            self.group_length = self.num_frames // self.num_groups
            self.fusion_all_group = MLP(input_dim = self.config.hidden_dim*self.group_length, 
               hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

            self.fusion_norm = FFN(d_model, dim_feedforward)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in x2ms_adapter.parameters(self):
            if x2ms_adapter.tensor_api.x2ms_dim(p) > 1:
                x2ms_adapter.nn_init.xavier_uniform_(p)

    def construct(self, src, pos=None):

        BS, N, C = src.shape
        if not pos is None:
            pos = x2ms_adapter.tensor_api.permute(pos, 1, 0, 2)
            
        if self.num_frames == 16:
            token_list = [x2ms_adapter.tensor_api.repeat(self.token[i:(i+1)], BS,1,1) for i in range(self.num_groups)]
            if self.sequence_stride ==1:
                src_groups = x2ms_adapter.tensor_api.chunk(x2ms_adapter.tensor_api.view(src, src.shape[0],src.shape[1]//self.num_groups ,-1), 4,dim=1)

            elif self.sequence_stride ==4:
                src_groups = []

                for i in range(self.num_groups):
                    groups = []
                    for j in range(self.group_length):
                        points_index_start = (i+j*self.sequence_stride)*self.num_proxy_points
                        points_index_end = points_index_start + self.num_proxy_points
                        groups.append(src[:,points_index_start:points_index_end])

                    groups = x2ms_adapter.cat(groups,-1)
                    src_groups.append(groups)

            else:
                raise NotImplementedError

            src_merge = x2ms_adapter.cat(src_groups,1)
            src = self.fusion_norm(src[:,:self.num_groups*self.num_proxy_points],self.fusion_all_group(src_merge))
            src = [x2ms_adapter.cat([token_list[i],src[:,i*self.num_proxy_points:(i+1)*self.num_proxy_points]],dim=1) for i in range(self.num_groups)]
            src = x2ms_adapter.cat(src,dim=0)

        else:
            token_list = [x2ms_adapter.tensor_api.repeat(self.token[i:(i+1)], BS,1,1) for i in range(self.num_groups)]
            src = [x2ms_adapter.cat([token_list[i],src[:,i*self.num_proxy_points:(i+1)*self.num_proxy_points]],dim=1) for i in range(self.num_groups)]
            src = x2ms_adapter.cat(src,dim=0)

        src = x2ms_adapter.tensor_api.permute(src, 1, 0, 2)
        memory,tokens = self.encoder(src,pos=pos) 

        memory = x2ms_adapter.cat(x2ms_adapter.tensor_api.chunk(memory[0:1], 4,dim=1),0)
        return memory, tokens
    

class TransformerEncoder(nn.Cell):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
        self.layers = x2ms_nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def construct(self, src,
                pos: Optional[x2ms_adapter.Tensor] = None):

        token_list = []
        output = src
        for layer in self.layers:
            output,tokens = layer(output,pos=pos)
            token_list.append(tokens)
        if self.norm is not None:
            output = x2ms_adapter.tensor_api.norm(self, output)

        return output,token_list


class TransformerEncoderLayer(nn.Cell):
    count = 0
    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count
        self.config = config
        self.num_point = num_points
        self.num_groups= num_groups
        self.self_attn = x2ms_nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = x2ms_nn.Linear(d_model, dim_feedforward)
        self.dropout = x2ms_nn.Dropout(dropout)
        self.linear2 = x2ms_nn.Linear(dim_feedforward, d_model)

        self.norm1 = x2ms_nn.LayerNorm(d_model)
        self.norm2 = x2ms_nn.LayerNorm(d_model)
        self.dropout1 = x2ms_nn.Dropout(dropout)
        self.dropout2 = x2ms_nn.Dropout(dropout)

        if self.layer_count <= self.config.enc_layers-1:
            self.cross_attn_layers = x2ms_nn.ModuleList()
            for _ in range(self.num_groups):
                self.cross_attn_layers.append(x2ms_nn.MultiheadAttention(d_model, nhead, dropout=dropout))

            self.ffn = FFN(d_model, dim_feedforward)
            self.fusion_all_groups = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)
    

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim,self.config.use_mlp_mixer.get('grid_size', 4),self.config.hidden_dim, self.config.use_mlp_mixer)


    def with_pos_embed(self, tensor, pos: Optional[x2ms_adapter.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[x2ms_adapter.Tensor] = None):

        src_intra_group_fusion = self.mlp_mixer_3d(src[1:])
        src = x2ms_adapter.cat([src[:1],src_intra_group_fusion],0)

        token = src[:1]

        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion

        src_summary = self.self_attn(token, key, value=src_intra_group_fusion)[0]
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        src = x2ms_adapter.cat([token,src[1:]],0)

        if self.layer_count <= self.config.enc_layers-1:
    
            src_all_groups = x2ms_adapter.tensor_api.view(src[1:], (src.shape[0]-1)*4,-1,src.shape[-1])
            src_groups_list = x2ms_adapter.tensor_api.chunk(src_all_groups, self.num_groups,0)

            src_all_groups = x2ms_adapter.cat(src_groups_list,-1)
            src_all_groups_fusion = self.fusion_all_groups(src_all_groups)

            key = self.with_pos_embed(src_all_groups_fusion, pos[1:])
            query_list = [self.with_pos_embed(query, pos[1:]) for query in src_groups_list]

            inter_group_fusion_list = []
            for i in range(self.num_groups):
                inter_group_fusion = self.cross_attn_layers[i](query_list[i], key, value=src_all_groups_fusion)[0]
                inter_group_fusion = self.ffn(src_groups_list[i],inter_group_fusion)
                inter_group_fusion_list.append(inter_group_fusion)

            src_inter_group_fusion = x2ms_adapter.cat(inter_group_fusion_list,1)

            src = x2ms_adapter.cat([src[:1],src_inter_group_fusion],0)

        return src, x2ms_adapter.cat(x2ms_adapter.tensor_api.chunk(src[:1], 4,1),0)

    def forward_pre(self, src,
                    pos: Optional[x2ms_adapter.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def construct(self, src,
                pos: Optional[x2ms_adapter.Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src,  pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return x2ms_adapter.nn_functional.relu
    if activation == "gelu":
        return x2ms_adapter.nn_functional.gelu
    if activation == "glu":
        # return F.glu
        return mindspore.ops.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FFN(nn.Cell):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,dout=None,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Implementation of Feedforward model
        self.linear1 = x2ms_nn.Linear(d_model, dim_feedforward)
        self.dropout = x2ms_nn.Dropout(dropout)
        self.linear2 = x2ms_nn.Linear(dim_feedforward, d_model)

        self.norm2 = x2ms_nn.LayerNorm(d_model)
        self.norm3 = x2ms_nn.LayerNorm(d_model)
        self.dropout1 = x2ms_nn.Dropout(dropout)
        self.dropout2 = x2ms_nn.Dropout(dropout)
        self.dropout3 = x2ms_nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def construct(self, tgt,tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

def build_transformer(args):
    return Transformer(
        config = args,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        num_lidar_points = args.num_lidar_points,
        num_proxy_points = args.num_proxy_points,
        num_frames = args.num_frames,
        sequence_stride = args.get('sequence_stride',1),
        num_groups=args.num_groups,
    )

