//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_UTILS_HPP
#define VOLUMESLICER_CUDA_UTILS_HPP

#include <VolumeSlicer/helper.hpp>

inline void CreateCUDATexture1D(int width,cudaArray** ppCudaArray,cudaTextureObject_t* pCudaTextureObject){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_RUNTIME_API_CALL(cudaMallocArray(ppCudaArray, &channelDesc, width));
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = *ppCudaArray;
    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true; // access with normalized texture coordinates
    texDescr.filterMode       = cudaFilterModeLinear; // linear interpolation
    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.readMode = cudaReadModeElementType;
    CUDA_RUNTIME_API_CALL(cudaCreateTextureObject(pCudaTextureObject, &texRes, &texDescr, NULL));
}

inline void CreateCUDATexture2D(int width,int height,cudaArray** ppCudaArray,cudaTextureObject_t* pCudaTextureObject){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_RUNTIME_API_CALL(cudaMallocArray(ppCudaArray, &channelDesc, width, height));
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = *ppCudaArray;
    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true; // access with normalized texture coordinates
    texDescr.filterMode       = cudaFilterModeLinear; // linear interpolation
    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    CUDA_RUNTIME_API_CALL(cudaCreateTextureObject(pCudaTextureObject, &texRes, &texDescr, NULL));
}

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

inline void UpdateCUDATexture1D(uint8_t* data,cudaArray* pCudaArray,uint32_t x_length,uint32_t x_offset){
//    CUDA_DRIVER_API_CALL(cuMemcpyHtoA((CUarray)pCudaArray,x_offset,data,x_length));
    CUDA_RUNTIME_API_CALL(cudaMemcpyToArray(pCudaArray,x_offset,0,data,x_length,cudaMemcpyHostToDevice));
}
inline void UpdateCUDATexture2D(uint8_t* data,cudaArray* pCudaArray,uint32_t x_length,uint32_t y_length,uint32_t x_offset,uint32_t y_offset){
    CUDA_MEMCPY2D m={0};
    m.srcMemoryType=CU_MEMORYTYPE_HOST;
    m.srcHost=data;

    m.dstMemoryType=CU_MEMORYTYPE_ARRAY;
    m.dstArray=(CUarray)pCudaArray;
    m.dstXInBytes=x_offset;
    m.dstY=y_offset;

    m.WidthInBytes=x_length;
    m.Height=y_length;

    CUDA_DRIVER_API_CALL(cuMemcpy2D(&m));
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
inline void UpdateCUDATexture3D(uint8_t* data,cudaArray* pCudaArray,uint32_t x_length,uint32_t y_length,uint32_t z_length,uint32_t x_offset,uint32_t y_offset,uint32_t z_offset){
    CUDA_MEMCPY3D m={0};
    m.srcMemoryType=CU_MEMORYTYPE_HOST;
    m.srcHost=data;

    m.dstMemoryType=CU_MEMORYTYPE_ARRAY;
    m.dstArray=(CUarray)pCudaArray;
    m.dstXInBytes=x_offset;
    m.dstY=y_offset;
    m.dstZ=z_offset;

    m.WidthInBytes=x_length;
    m.Height=y_length;
    m.Depth=z_length;

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
