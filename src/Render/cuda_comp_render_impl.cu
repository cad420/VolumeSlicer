//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP

#include "cuda_comp_render_impl.cuh"
#include "Algorithm/helper_math.h"
using namespace CUDARenderer;

namespace {
    __constant__ CUDACompRenderParameter cudaCompRenderParameter;
    __constant__ CompVolumeParameter compVolumeParameter;
    __constant__ uint4 *mappingTable;
    __constant__ uint lodMappingTableOffset[10];
    __constant__ cudaTextureObject_t cacheVolumes[10];
    __constant__ cudaTextureObject_t transferFunc;
    __constant__ cudaTextureObject_t preIntTransferFunc;
    __constant__ uint *image;

    uint *d_image = nullptr;
    int image_w = 0, image_h = 0;
    uint4 *d_mappingTable = nullptr;


    __device__ void VirtualSample(){

    }

    __global__ void CUDARenderKernel(){

    }
}
void CUDARenderer::UploadTransferFunc(float *data, size_t size) {

}

void CUDARender(uint32_t *image) {
}

void UploadLightParameter(const LightParameter &) {

}

void UploadCompVolumeParameter(const CompVolumeParameter &) {

}

void UploadCUDACompRenderParameter(const CUDACompRenderParameter &) {

}

void SetCUDATextureObject(cudaTextureObject_t *textures, size_t size) {

}

void UploadPreIntTransferFunc(float *data, size_t size) {

}

#endif //VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
