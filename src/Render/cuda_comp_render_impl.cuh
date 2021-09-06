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
        float step;//0.0001
        int steps;
        float3 view_direction;
        float3 up;
        float3 right;
        float3 space;//0.00032 0.00032 0.001 um
    };

//not changed while start rendering
    struct CompVolumeParameter{
        int min_lod,max_lod;
        int block_length;
        int padding;
        int no_padding_block_length;
        int4 texture_shape;
        int3 block_dim;//lod 0
        int3 volume_board;//47*508 59*508 21*508
    };

    struct LightParameter{
        float ka;
        float kd;
        float ks;
        float shininess;
        float4 bg_color;
    };

    void UploadMappingTable(const uint32_t* data,size_t size);

    void UploadLodMappingTableOffset(const uint32_t* data,size_t size);

    void UploadTransferFunc(float* data,size_t size=256);

    void UploadPreIntTransferFunc(float* data,size_t size=65536);

    void SetCUDATextureObject(cudaTextureObject_t* textures,size_t size);

    void UploadCUDACompRenderParameter(const CUDACompRenderParameter&);

    void UploadCompVolumeParameter(const CompVolumeParameter&);

    void UploadLightParameter(const LightParameter&);

    void CUDACalcBlock(uint32_t* missed_blocks,size_t size,uint32_t w,uint32_t h);

    void CUDARender(uint32_t w,uint32_t h,uint32_t* image);
}


#endif //VOLUMESLICER_CUDA_COMP_RENDER_IMPL_CUH
