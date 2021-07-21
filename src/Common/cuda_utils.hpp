//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_UTILS_HPP
#define VOLUMESLICER_CUDA_UTILS_HPP

#include <VolumeSlicer/helper.hpp>


//only create uint8_t 3D CUDA Texture
inline void CreateCUDATexture3D(cudaExtent textureSize, cudaArray **ppCudaArray, cudaTextureObject_t *pCudaTextureObject){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
    CUDA_RUNTIME_API_CALL(cudaMalloc3DArray(ppCudaArray, &channelDesc, textureSize));
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = *ppCudaArray;
    cudaTextureDesc             texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = true; // access with normalized texture coordinates
    texDesc.filterMode       = cudaFilterModeLinear; // linear interpolation
    texDesc.borderColor[0]=0.f;
    texDesc.borderColor[1]=0.f;
    texDesc.borderColor[2]=0.f;
    texDesc.borderColor[3]=0.f;
    texDesc.addressMode[0]=cudaAddressModeBorder;
    texDesc.addressMode[1]=cudaAddressModeBorder;
    texDesc.addressMode[2]=cudaAddressModeBorder;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    CUDA_RUNTIME_API_CALL(cudaCreateTextureObject(pCudaTextureObject, &texRes, &texDesc, NULL));
}

//offset if count by byte
inline void UpdateCUDATexture3D(uint8_t* data,cudaArray* pCudaArray,uint32_t block_length,uint32_t x_offset,uint32_t y_offset,uint32_t z_offset){
    CUDA_MEMCPY3D m={0};
    m.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    m.srcDevice=(CUdeviceptr)data;

    m.dstMemoryType=CU_MEMORYTYPE_ARRAY;
    m.dstArray=(CUarray)pCudaArray;
    m.dstXInBytes=x_offset;
    m.dstY=y_offset;
    m.dstZ=z_offset;

    m.WidthInBytes=block_length;
    m.Height=block_length;
    m.Depth=block_length;

    CUDA_DRIVER_API_CALL(cuMemcpy3D(&m));
}


__host__ __device__ inline int PowII(int x,int y){
    int res=1;
    for(int i=0;i<y;i++){
        res*=x;
    }
    return res;
}

#endif //VOLUMESLICER_CUDA_UTILS_HPP
