from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='deep_point',
    version='v1.0',
    description='deep layers used to convert between point and voxels',
    install_requires=['PyYAML>=5.4.1', 'scipy>=1.3.1'],
    ext_modules=[
        CppExtension(name = 'point_deep.cpu_kernel',
                    sources = ['pytorch_lib/src/point_deep.cpp']),
        CUDAExtension(name = 'point_deep.cuda_kernel',
                    sources = ['pytorch_lib/src/point_deep_cuda.cpp', 'pytorch_lib/src/point_deep_cuda_kernel.cu'],
                    include_dirs = ['pytorch_lib/src']),
    ],
    cmdclass={'build_ext': BuildExtension}
)