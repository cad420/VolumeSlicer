//
// Created by wyz on 2021/9/13.
//

#pragma once

#include <VolumeSlicer/CUDA/cuda_context.hpp>
#include <VolumeSlicer/Utils/box.hpp>

#include "Common/helper_math.h"

#include <unordered_set>

using namespace vs;

namespace CUDAOffRenderer
{

struct CUDAOffCompRenderPolicy
{
    double lod_dist[10];
};

struct CUDAOffCompRenderParameter
{

    int image_w;
    int image_h;
    float3 camera_pos;
    float3 right;
    float3 up;
    float3 front;
    double fov;
    float step;
};

struct CompVolumeParameter
{
    float3 volume_board; // volume_dim * volume_space
    float3 volume_space;
    int3 volume_dim;
    int3 block_dim;
    float voxel;
    int block_length;
    int padding;
    int no_padding_block_length;
    int min_lod, max_lod;
    int4 volume_texture_shape;
};

struct ShadingParameter
{
    float ka;
    float ks;
    float kd;
    float shininess;
};

void UploadTransferFunc(float *data, size_t size = 256);

void UploadPreIntTransferFunc(float *data, size_t size = 65536);

void UploadCUDAOffCompRenderPolicy(const CUDAOffCompRenderPolicy &);

void UploadCUDAOffCompRenderParameter(const CUDAOffCompRenderParameter &);

void UploadCompVolumeParameter(const CompVolumeParameter &);

void UploadShadingParameter(const ShadingParameter &);

void UploadMappingTable(const uint32_t *data, size_t size);

void UploadLodMappingTableOffset(const uint32_t *data, size_t size);

void SetCUDATextureObject(cudaTextureObject_t *textures, size_t size);

void CUDARenderPrepare(int w, int h);

void CUDARender(std::unordered_set<int4, Hash_Int4> &missed_blocks);

void GetRenderImage(uint8_t *data);

} // namespace CUDAOffRenderer