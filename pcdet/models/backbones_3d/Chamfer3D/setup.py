from setuptools import setup
import mindspore
import x2ms_adapter

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(x2ms_adapter.tensor_api.split(__file__, '/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(x2ms_adapter.tensor_api.split(__file__, '/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })