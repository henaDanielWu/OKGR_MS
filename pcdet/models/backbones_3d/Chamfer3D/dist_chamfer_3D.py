import importlib
import os
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
# chamfer_found = importlib.find_loader("chamfer_3D") is not None
# if not chamfer_found:
#     ## Cool trick from https://github.com/chrdiller
#     # print("Jitting Chamfer 3D")
#
#     chamfer_3D = load(name="chamfer_3D",
#           sources=[
#               "/".join(x2ms_adapter.tensor_api.split(os.path.abspath(__file__), '/')[:-1] + ["chamfer_cuda.cpp"]),
#               "/".join(x2ms_adapter.tensor_api.split(os.path.abspath(__file__), '/')[:-1] + ["chamfer3D.cu"]),
#               ])
#     print("Loaded JIT 3D CUDA chamfer distance")
#
# else:
from . import chamfer_3D
    # print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, xyz1, xyz2):
        batchsize, n, _ = x2ms_adapter.tensor_api.x2ms_size(xyz1)
        _, m, _ = x2ms_adapter.tensor_api.x2ms_size(xyz2)
        device = xyz1.device

        dist1 = x2ms_adapter.zeros(batchsize, n)
        dist2 = x2ms_adapter.zeros(batchsize, m)

        idx1 = x2ms_adapter.tensor_api.x2ms_type(x2ms_adapter.zeros(batchsize, n), x2ms_adapter.IntTensor)
        idx2 = x2ms_adapter.tensor_api.x2ms_type(x2ms_adapter.zeros(batchsize, m), x2ms_adapter.IntTensor)

        dist1 = x2ms_adapter.to(dist1, device)
        dist2 = x2ms_adapter.to(dist2, device)
        idx1 = x2ms_adapter.to(idx1, device)
        idx2 = x2ms_adapter.to(idx2, device)
        x2ms_adapter.cuda_set_device(device)
        print("chamfer_3D")
        chamfer_3D.forward( xyz1, xyz2, dist1, dist2, idx1, idx2)
        # ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = x2ms_adapter.tensor_api.contiguous(graddist1)
        graddist2 = x2ms_adapter.tensor_api.contiguous(graddist2)
        device = graddist1.device

        gradxyz1 = x2ms_adapter.zeros(x2ms_adapter.tensor_api.x2ms_size(xyz1))
        gradxyz2 = x2ms_adapter.zeros(x2ms_adapter.tensor_api.x2ms_size(xyz2))

        gradxyz1 = x2ms_adapter.to(gradxyz1, device)
        gradxyz2 = x2ms_adapter.to(gradxyz2, device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Cell):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def construct(self, input1, input2):
        input1 = x2ms_adapter.tensor_api.contiguous(input1)
        input2 = x2ms_adapter.tensor_api.contiguous(input2)
        return x2ms_adapter.nn_cell.apply(chamfer_3DFunction, input1, input2)

