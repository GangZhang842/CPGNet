#include <torch/extension.h>
#include <vector>
#include <cmath>

#ifdef VERSION_GE_1_3
#define DATA_PTR data_ptr
#else
#define DATA_PTR data
#endif

// maxpool
namespace maxpool{
    // voxel max pooling forward
    // pcds_feat, (BS, C, N, 1)
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // voxel_out, (BS, C, D1, D2, ..., Dn)
    // voxel_max_idx, (BS, N)
    template<typename real>
    void VoxelMaxPoolUpdateOutputInit(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data,
                                    int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                    int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        int64_t bs, c, n;
        int64_t index_pcds, index_ind, index_voxel;
        int64_t index_res;
        for(int64_t i=0; i < loop; i++){
            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_ind = bs * N * D + n * D;
            index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1]; // bs and c

            bool flag = true;
            for(int64_t d=0; d < D; d++){
                int64_t ind_tmp = int64_t(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if((ind_tmp >=0) && (ind_tmp < output_size[d])){
                    index_voxel = index_voxel + ind_tmp * voxel_out_stride[2 + d];
                }
                else{
                    flag = false;
                }
            }

            if(flag){
                voxel_out_data[index_voxel] = pcds_feat_data[index_pcds];
            }
        }
    }

    template<typename real>
    void VoxelMaxPoolUpdateOutputKernel(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data,
                                        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        int64_t bs, c, n;
        int64_t index_pcds, index_ind, index_voxel;
        int64_t index_res;
        for(int64_t i=0; i < loop; i++){
            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_ind = bs * N * D + n * D;
            index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1]; // bs and c

            bool flag = true;
            for(int64_t d=0; d < D; d++){
                int64_t ind_tmp = int64_t(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if((ind_tmp >=0) && (ind_tmp < output_size[d])){
                    index_voxel = index_voxel + ind_tmp * voxel_out_stride[2 + d];
                }
                else{
                    flag = false;
                }
            }

            if(flag){
                if(voxel_out_data[index_voxel] < pcds_feat_data[index_pcds]){
                    voxel_out_data[index_voxel] = pcds_feat_data[index_pcds];
                }
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
    void VoxelMaxPoolUpdateBackwardKernel(real* pcds_feat_data, real* pcds_ind_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data,
                                        int64_t BS, int64_t C, int64_t N, int64_t D, int64_t loop,
                                        int64_t* voxel_out_size, int64_t* voxel_out_stride, int64_t* output_size, float* scale_rate)
    {
        int64_t bs, c, n;
        int64_t index_pcds, index_ind, index_voxel;
        int64_t index_res;
        for(int64_t i=0; i < loop; i++){
            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_ind = bs * N * D + n * D;
            index_voxel = bs * voxel_out_stride[0] + c * voxel_out_stride[1]; // bs and c

            bool flag = true;
            for(int64_t d=0; d < D; d++){
                int64_t ind_tmp = int64_t(static_cast<float>(pcds_ind_data[index_ind + d]) * scale_rate[d]);
                if((ind_tmp >=0) && (ind_tmp < output_size[d])){
                    index_voxel = index_voxel + ind_tmp * voxel_out_stride[2 + d];
                }
                else{
                    flag = false;
                }
            }

            if(flag){
                if(voxel_out_data[index_voxel] == pcds_feat_data[index_pcds]){
                    grad_pcds_feat_data[index_pcds] = grad_voxel_out_data[index_voxel];
                }
            }
        }
    }
}

void voxel_maxpooling_cpu_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    int64_t BS = pcds_feat.size(0);
    int64_t C = pcds_feat.size(1);
    int64_t N = pcds_feat.size(2);

    int64_t D = pcds_ind.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pcds_feat.scalar_type(), "VoxelMaxPoolUpdateOutputInit", [&] {
        scalar_t *pcds_feat_data = pcds_feat.DATA_PTR<scalar_t>();
        scalar_t *pcds_ind_data = pcds_ind.DATA_PTR<scalar_t>();
        scalar_t *voxel_out_data = voxel_out.DATA_PTR<scalar_t>();

        maxpool::VoxelMaxPoolUpdateOutputInit<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, BS, C, N, D, loop,
        voxel_out_size.DATA_PTR<int64_t>(), voxel_out_stride.DATA_PTR<int64_t>(), output_size.DATA_PTR<int64_t>(), scale_rate.DATA_PTR<float>());

        maxpool::VoxelMaxPoolUpdateOutputKernel<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, BS, C, N, D, loop,
        voxel_out_size.DATA_PTR<int64_t>(), voxel_out_stride.DATA_PTR<int64_t>(), output_size.DATA_PTR<int64_t>(), scale_rate.DATA_PTR<float>());
    });
}


void voxel_maxpooling_cpu_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    int64_t BS = pcds_feat.size(0);
    int64_t C = pcds_feat.size(1);
    int64_t N = pcds_feat.size(2);

    int64_t D = pcds_ind.size(2);
    int64_t loop = BS * C * N;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pcds_feat.scalar_type(), "VoxelMaxPoolUpdateBackwardKernel", [&] {
        scalar_t *pcds_feat_data = pcds_feat.DATA_PTR<scalar_t>();
        scalar_t *pcds_ind_data = pcds_ind.DATA_PTR<scalar_t>();
        scalar_t *voxel_out_data = voxel_out.DATA_PTR<scalar_t>();

        scalar_t *grad_pcds_feat_data = grad_pcds_feat.DATA_PTR<scalar_t>();
        scalar_t *grad_voxel_out_data = grad_voxel_out.DATA_PTR<scalar_t>();

        maxpool::VoxelMaxPoolUpdateBackwardKernel<scalar_t>(pcds_feat_data, pcds_ind_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data,
        BS, C, N, D, loop, voxel_out_size.DATA_PTR<int64_t>(), voxel_out_stride.DATA_PTR<int64_t>(), output_size.DATA_PTR<int64_t>(), scale_rate.DATA_PTR<float>());
    });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_maxpooling_cpu_forward", &voxel_maxpooling_cpu_forward, "maxpooling forward (CPU)");
  m.def("voxel_maxpooling_cpu_backward", &voxel_maxpooling_cpu_backward, "maxpooling backward (CPU)");
}