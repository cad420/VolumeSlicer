//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_COMP_RENDER_IMPL_CUH
#define VOLUMESLICER_CUDA_COMP_RENDER_IMPL_CUH
#include <VolumeSlicer/cuda_context.hpp>
namespace CUDARenderer{
    struct CUDACompRenderParameter{
        float3 view_pos;
        int w,h;
        float fov;
        float step;
        float3 view_direction;
        float3 up;
        float3 right;
        float3 space;
    };

//not changed while start rendering
    struct CompVolumeParameter{
        int min_lod,max_lod;
        int block_length;
        int padding;
        int no_padding_block_length;
        int3 block_dim;//lod 0
        int3 texture_shape;
        int3 volume_board;
    };

    struct LightParameter{
        float ka;
        float kd;
        float ks;
        float shininess;
        float4 bg_color;
    };

    void UploadTransferFunc(float* data,size_t size);

    void UploadPreIntTransferFunc(float* data,size_t size);

    void SetCUDATextureObject(cudaTextureObject_t* textures,size_t size);

    void UploadCUDACompRenderParameter(const CUDACompRenderParameter&);

    void UploadCompVolumeParameter(const CompVolumeParameter&);

    void UploadLightParameter(const LightParameter&);

    void CUDARender(uint32_t* image);
}


#endif //VOLUMESLICER_CUDA_COMP_RENDER_IMPL_CUH
