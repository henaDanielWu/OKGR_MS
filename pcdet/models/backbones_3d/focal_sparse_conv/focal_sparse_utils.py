import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


class FocalLoss(nn.Cell):

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def one_hot(self, index, classes):
        size = x2ms_adapter.tensor_api.x2ms_size(index) + (classes,)
        view = x2ms_adapter.tensor_api.x2ms_size(index) + (1,)

        mask = x2ms_adapter.to(x2ms_adapter.tensor_api.fill_(x2ms_adapter.Tensor(*size), 0), index.device)

        index = x2ms_adapter.tensor_api.view(index, *view)
        ones = 1.

        if isinstance(index, x2ms_adapter.autograd.Variable):
            ones = x2ms_adapter.autograd.Variable(x2ms_adapter.to(x2ms_adapter.tensor_api.fill_(x2ms_adapter.Tensor(x2ms_adapter.tensor_api.x2ms_size(index)), 1), index.device))
            mask = x2ms_adapter.autograd.Variable(mask, volatile=index.volatile)

        return x2ms_adapter.tensor_api.scatter_(mask, 1, index, ones)

    def construct(self, input, target):
        y = self.one_hot(target, x2ms_adapter.tensor_api.x2ms_size(input, -1))
        logit = x2ms_adapter.nn_functional.softmax(input, dim=-1)
        logit = x2ms_adapter.tensor_api.clamp(logit, self.eps, 1. - self.eps)

        loss = -1 * y * x2ms_adapter.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return x2ms_adapter.tensor_api.x2ms_mean(loss)

def sort_by_indices(features, indices, features_add=None):
    """
        To sort the sparse features with its indices in a convenient manner.
        Args:
            features: [N, C], sparse features
            indices: [N, 4], indices of sparse features
            features_add: [N, C], additional features to sort
    """
    idx = indices[:, 1:]
    idx_sum = idx.select(1, 0) * x2ms_adapter.tensor_api.x2ms_max(idx[:, 1]) * x2ms_adapter.tensor_api.x2ms_max(idx[:, 2]) + idx.select(1, 1) * x2ms_adapter.tensor_api.x2ms_max(idx[:, 2]) + idx.select(1, 2)
    _, ind = x2ms_adapter.tensor_api.sort(idx_sum)
    features = features[ind]
    indices = indices[ind]
    if not features_add is None:
        features_add = features_add[ind]
    return features, indices, features_add

def check_repeat(features, indices, features_add=None, sort_first=True, flip_first=True):
    """
        Check that whether there are replicate indices in the sparse features, 
        remove the replicate features if any.
    """
    if sort_first:
        features, indices, features_add = sort_by_indices(features, indices, features_add)

    if flip_first:
        features, indices = x2ms_adapter.tensor_api.flip(features, [0]), x2ms_adapter.tensor_api.flip(indices, [0])

    if not features_add is None:
        features_add=x2ms_adapter.tensor_api.flip(features_add, [0])

    idx = x2ms_adapter.tensor_api.x2ms_int(indices[:, 1:])
    idx_sum = x2ms_adapter.add(x2ms_adapter.add(idx.select(1, 0) * x2ms_adapter.tensor_api.x2ms_max(idx[:, 1]) * x2ms_adapter.tensor_api.x2ms_max(idx[:, 2]), idx.select(1, 1) * x2ms_adapter.tensor_api.x2ms_max(idx[:, 2])), idx.select(1, 2))
    # _unique, inverse, counts = torch.unique_consecutive(idx_sum, return_inverse=True, return_counts=True, dim=0)
    _unique, inverse, counts = mindspore.ops.unique_consecutive(idx_sum, return_inverse=True, return_counts=True, dim=0)

    if _unique.shape[0] < indices.shape[0]:
        perm = x2ms_adapter.arange(x2ms_adapter.tensor_api.x2ms_size(inverse, 0), dtype=inverse.dtype, device=inverse.device)
        features_new = x2ms_adapter.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
        x2ms_adapter.tensor_api.index_add_(features_new, 0, x2ms_adapter.tensor_api.long(inverse), features)
        features = features_new
        perm_ = x2ms_adapter.tensor_api.scatter_(inverse.new_empty(x2ms_adapter.tensor_api.x2ms_size(_unique, 0)), 0, inverse, perm)
        indices = x2ms_adapter.tensor_api.x2ms_int(indices[perm_])

        if not features_add is None:
            features_add_new = x2ms_adapter.zeros((_unique.shape[0],), device=features_add.device)
            x2ms_adapter.tensor_api.index_add_(features_add_new, 0, x2ms_adapter.tensor_api.long(inverse), features_add)
            features_add = features_add_new / counts
    return features, indices, features_add


def split_voxels(x, b, imps_3d, voxels_3d, kernel_offsets, mask_multi=True, topk=True, threshold=0.5):
    """
        Generate and split the voxels into foreground and background sparse features, based on the predicted importance values.
        Args:
            x: [N, C], input sparse features
            b: int, batch size id
            imps_3d: [N, kernelsize**3], the prediced importance values
            voxels_3d: [N, 3], the 3d positions of voxel centers 
            kernel_offsets: [kernelsize**3, 3], the offset coords in an kernel
            mask_multi: bool, whether to multiply the predicted mask to features
            topk: bool, whether to use topk or threshold for selection
            threshold: float, threshold value
    """
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    mask_voxel = x2ms_adapter.tensor_api.sigmoid(imps_3d[batch_index, -1])
    mask_kernel = x2ms_adapter.tensor_api.sigmoid(imps_3d[batch_index, :-1])

    if mask_multi:
        features_ori *= x2ms_adapter.tensor_api.unsqueeze(mask_voxel, -1)

    if topk:
        _, indices = x2ms_adapter.tensor_api.sort(mask_voxel, descending=True)
        indices_fore = indices[:int(mask_voxel.shape[0]*threshold)]
        indices_back = indices[int(mask_voxel.shape[0]*threshold):]
    else:
        indices_fore = mask_voxel > threshold
        indices_back = mask_voxel <= threshold

    features_fore = features_ori[indices_fore]
    coords_fore = indices_ori[indices_fore]

    mask_kernel_fore = mask_kernel[indices_fore]
    mask_kernel_bool = mask_kernel_fore>=threshold
    voxel_kerels_imp = x2ms_adapter.tensor_api.repeat(x2ms_adapter.tensor_api.unsqueeze(kernel_offsets, 0), mask_kernel_bool.shape[0],1, 1)
    mask_kernel_fore = mask_kernel[indices_fore][mask_kernel_bool]
    indices_fore_kernels = x2ms_adapter.tensor_api.repeat(x2ms_adapter.tensor_api.unsqueeze(coords_fore[:, 1:], 1), 1, kernel_offsets.shape[0], 1)
    indices_with_imp = indices_fore_kernels + voxel_kerels_imp
    selected_indices = indices_with_imp[mask_kernel_bool]
    spatial_indices = (selected_indices[:, 0] >0) * (selected_indices[:, 1] >0) * (selected_indices[:, 2] >0)  * \
                        (selected_indices[:, 0] < x.spatial_shape[0]) * (selected_indices[:, 1] < x.spatial_shape[1]) * (selected_indices[:, 2] < x.spatial_shape[2])
    selected_indices = selected_indices[spatial_indices]
    mask_kernel_fore = mask_kernel_fore[spatial_indices]
    selected_indices = x2ms_adapter.cat([x2ms_adapter.ones((selected_indices.shape[0], 1), device=features_fore.device)*b, selected_indices], dim=1)

    selected_features = x2ms_adapter.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_fore.device)

    features_fore_cat = x2ms_adapter.cat([features_fore, selected_features], dim=0)
    coords_fore = x2ms_adapter.cat([coords_fore, selected_indices], dim=0)
    mask_kernel_fore = x2ms_adapter.cat([x2ms_adapter.ones(features_fore.shape[0], device=features_fore.device), mask_kernel_fore], dim=0)

    features_fore, coords_fore, mask_kernel_fore = check_repeat(features_fore_cat, coords_fore, features_add=mask_kernel_fore)

    features_back = features_ori[indices_back]
    coords_back = indices_ori[indices_back]

    return features_fore, coords_fore, features_back, coords_back, mask_kernel_fore
