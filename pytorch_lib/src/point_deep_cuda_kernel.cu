#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "atomics.cuh"

template<typename scalar_t>
inline scalar_t* DATA_PTR(at::Tensor mat){
    return at::cuda::detail::getTensorInfo<scalar_t, int64_t>(mat).data;
}

// maxpool
namespace maxpool{
    // voxel max pooling forward
    // pcds_feat, (BS, C, N, 1)
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // voxel_out, (BS, C, D1, D2, ..., Dn)
    // voxel_max_idx, (BS, N)
    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputComputeIdx(real* pcds_ind_data, int64_t* voxel_max_idx_data,
                                                    int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                                    int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            int64_t bs, n;
            int64_t index_ind, index_voxel;

            bs = i / N;
            n = i - bs * N;

            index_ind = i * D;
            index_voxel = bs * voxel_out_stride[0]; // bs and c=0

            int flag = 1;
            for(int64_t d=0; d < D; d++){
                int64_t ind_tmp = int64_t(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if((ind_tmp >=0) && (ind_tmp < output_size[d])){
                    index_voxel = index_voxel + ind_tmp * voxel_out_stride[2 + d];
                }
                else{
                    flag = 0;
                }
            }
            
            if(flag == 1){
                voxel_max_idx_data[i] = index_voxel;
            }
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputInit(real* pcds_feat_data, real* voxel_out_data, int64_t* voxel_max_idx_data,
                                                int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                                int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size)
    {
        for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            int64_t bs, c, n;
            int64_t index_pcds, index_voxel0;
            int64_t index_res;

            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_voxel0 = voxel_max_idx_data[bs * N + n];
            if(index_voxel0 >= 0){
                voxel_out_data[index_voxel0 + c * voxel_out_stride[1]] = pcds_feat_data[index_pcds];
            }
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputKernel(real* pcds_feat_data, real* voxel_out_data, int64_t* voxel_max_idx_data,
                                                int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                                int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size)
    {
        for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            int64_t bs, c, n;
            int64_t index_pcds, index_voxel0;
            int64_t index_res;

            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_voxel0 = voxel_max_idx_data[bs * N + n];
            if(index_voxel0 >= 0){
                atomMax(&voxel_out_data[index_voxel0 + c * voxel_out_stride[1]], pcds_feat_data[index_pcds]);
            }
        }
    }

    // backward
    // pcds_feat, (BS, C, N, 1)
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // voxel_out, (BS, C, D1, D2, ..., Dn)
    // voxel_max_idx, (BS, N)
    // grad_pcds_feat, (BS, C, N, 1)
    // grad_voxel_out, (BS, C, D1, D2, ..., Dn)
    template<typename real>
    __global__ void VoxelMaxPoolUpdateBackwardKernel(real* pcds_feat_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data, int64_t* voxel_max_idx_data,
                                                    int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                                    int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size)
    {
        for(int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            int64_t bs, c, n;
            int64_t index_pcds, index_voxel0;
            int64_t index_res;

            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_voxel0 = voxel_max_idx_data[bs * N + n];
            if(index_voxel0 >= 0){
                int64_t index_voxel = index_voxel0 + c * voxel_out_stride[1];
                if(voxel_out_data[index_voxel] == pcds_feat_data[index_pcds]){
                    grad_pcds_feat_data[index_pcds] = grad_voxel_out_data[index_voxel];
                }
            }
        }
    }
}

void voxel_maxpooling_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C = pcds_feat.size(1);
    int64_t N = pcds_feat.size(2);
    int64_t D = pcds_ind.size(2);

    int64_t loop1 = BS * N;
    int64_t loop2 = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pcds_feat.scalar_type(), "VoxelMaxPoolUpdateOutputComputeIdx", [&] {
        scalar_t *pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        scalar_t *pcds_ind_data = DATA_PTR<scalar_t>(pcds_ind);
        scalar_t *voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        int64_t *voxel_max_idx_data = DATA_PTR<int64_t>(voxel_max_idx);

        maxpool::VoxelMaxPoolUpdateOutputComputeIdx<scalar_t><<<BLOCKS(loop1), THREADS>>>(pcds_ind_data, voxel_max_idx_data, BS, C, N, D, loop1,
        DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size), DATA_PTR<float>(scale_rate));

        maxpool::VoxelMaxPoolUpdateOutputInit<scalar_t><<<BLOCKS(loop2), THREADS>>>(pcds_feat_data, voxel_out_data, voxel_max_idx_data, BS, C, N, D, loop2,
        DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size));

        maxpool::VoxelMaxPoolUpdateOutputKernel<scalar_t><<<BLOCKS(loop2), THREADS>>>(pcds_feat_data, voxel_out_data, voxel_max_idx_data, BS, C, N, D, loop2,
        DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size));
    });
}

void voxel_maxpooling_cuda_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int64_t BS = pcds_feat.size(0);
    int64_t C = pcds_feat.size(1);
    int64_t N = pcds_feat.size(2);
    int64_t D = pcds_ind.size(2);

    int64_t loop = BS * C * N;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pcds_feat.scalar_type(), "VoxelMaxPoolUpdateBackwardKernel", [&] {
        scalar_t *pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        scalar_t *pcds_ind_data = DATA_PTR<scalar_t>(pcds_ind);
        scalar_t *voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        int64_t* voxel_max_idx_data = DATA_PTR<int64_t>(voxel_max_idx);

        scalar_t *grad_pcds_feat_data = DATA_PTR<scalar_t>(grad_pcds_feat);
        scalar_t *grad_voxel_out_data = DATA_PTR<scalar_t>(grad_voxel_out);

        maxpool::VoxelMaxPoolUpdateBackwardKernel<scalar_t><<<BLOCKS(loop), THREADS>>>(pcds_feat_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data, voxel_max_idx_data,
        BS, C, N, D, loop, DATA_PTR<int64_t>(voxel_out_size), DATA_PTR<int64_t>(voxel_out_stride), DATA_PTR<int64_t>(output_size));
    });
}