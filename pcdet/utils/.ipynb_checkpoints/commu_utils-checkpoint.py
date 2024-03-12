"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.

deeply borrow from maskrcnn-benchmark and ST3D
"""

import pickle
import time
import mindspore
import x2ms_adapter
import x2ms_adapter.torch_api.distributed as x2ms_distributed
import x2ms_adapter.torch_api.torch_base_api as x2ms_base


def get_world_size():
    if not x2ms_distributed.is_available():
        return 1
    if not x2ms_distributed.is_initialized():
        return 1
    return x2ms_distributed.cuda_device_count()


def get_rank():
    if not x2ms_distributed.is_available():
        return 0
    if not x2ms_distributed.is_initialized():
        return 0
    return x2ms_distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not x2ms_distributed.is_available():
        return
    if not x2ms_distributed.is_initialized():
        return
    world_size = x2ms_distributed.cuda_device_count()
    if world_size == 1:
        return
    x2ms_distributed.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, mindspore.Tensor):
        buffer = pickle.dumps(data)
        storage = x2ms_base.ByteStorage.from_buffer(buffer)
        tensor = x2ms_adapter.to(x2ms_adapter.ByteTensor(storage), "cuda")
    else:
        origin_size = x2ms_adapter.tensor_api.x2ms_size(data)
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = x2ms_adapter.to(x2ms_adapter.LongTensor([x2ms_adapter.tensor_api.numel(tensor)]), "cuda")
    size_list = [x2ms_adapter.to(x2ms_adapter.LongTensor([0]), "cuda") for _ in range(world_size)]
    x2ms_distributed.all_gather(size_list, local_size)
    size_list = [int(x2ms_adapter.tensor_api.item(size)) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(x2ms_adapter.to(x2ms_adapter.FloatTensor(size=(max_size,)), tensor_type))
    if local_size != max_size:
        padding = x2ms_adapter.to(x2ms_adapter.FloatTensor(size=(max_size - local_size,)), tensor_type)
        tensor = x2ms_adapter.cat((tensor, padding), dim=0)
    x2ms_distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = x2ms_adapter.tensor_api.numpy(tensor).tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    names = []
    values = []
    # sort the keys so that they are consistent across processes
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = x2ms_adapter.stack(values, dim=0)
    x2ms_distributed.reduce(values, dst=0)
    if x2ms_distributed.get_rank() == 0 and average:
        # only main process gets accumulated, so only divide by
        # world_size in this case
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def average_reduce_value(data):
    data_list = all_gather(data)
    return sum(data_list) / len(data_list)


def all_reduce(data, op="sum", average=False):

    def op_map(op):
        op_dict = {
            "SUM": mindspore.ops.ReduceOp.SUM,
            "MAX": mindspore.ops.ReduceOp.MAX,
            "MIN": mindspore.ops.ReduceOp.MIN,
            "PRODUCT": mindspore.ops.ReduceOp.PROD,
        }
        return op_dict[op]

    world_size = get_world_size()
    if world_size > 1:
        reduced_data = x2ms_adapter.tensor_api.clone(data)
        x2ms_distributed.all_reduce(reduced_data, op=op_map(op.upper()))
        if average:
            assert op.upper() == 'SUM'
            return reduced_data / world_size
        else:
            return reduced_data
    return data


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [x2ms_adapter.ones_like(tensor)
        for _ in range(x2ms_distributed.cuda_device_count())]
    x2ms_distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = x2ms_adapter.cat(tensors_gather, dim=0)
    return output
