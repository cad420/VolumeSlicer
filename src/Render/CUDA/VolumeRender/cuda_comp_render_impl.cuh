//
// Created by wyz on 2021/7/21.
//

#pragma once

#include <VolumeSlicer/CUDA/cuda_context.hpp>
#include <VolumeSlicer/Render/render.hpp>

using namespace vs;

namespace CUDARenderer{

    struct CUDACompRenderPolicy{
        double lod_dist[10];
    };

    /**
     * @brief Maybe changed during every frame render
     */
    struct CUDACompRenderParameter{
        float3 view_pos;
        int w,h;
        float fov;
        float step;//according to space
        int steps;
        float3 view_direction;
        float3 up;
        float3 right;
        float3 space;//um
        bool mpi_render;
    };

    /**
     * @brief Parameters that are not changed while start rendering
     */
    struct CompVolumeParameter{
        int cdf_block_num;
        int cdf_dim_len;//xyz is all the same
        int cdf_block_length;
        int min_lod,max_lod;
        int block_length;
        int padding;
        int no_padding_block_length;
        int4 texture_shape;
        int3 block_dim;//lod 0
        float3 volume_board;//
        int3 volume_dim;//in voxel
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

    void UploadCUDACompRenderPolicy(const CUDACompRenderPolicy&);

    void UploadCUDACompRenderParameter(const CUDACompRenderParameter&);

    void UploadMPIRenderParameter(const MPIRenderParameter& mpi_render);

    void UploadCompVolumeParameter(const CompVolumeParameter&);

    void UploadLightParameter(const LightParameter&);

    void CUDACalcBlock(uint32_t* missed_blocks,size_t size,uint32_t w,uint32_t h);

    void CUDARender(uint32_t w,uint32_t h,uint8_t * image);

    void UploadCDFMap(const uint32_t** data,int n,size_t* size);

    void DeleteAllCUDAResources();
}



