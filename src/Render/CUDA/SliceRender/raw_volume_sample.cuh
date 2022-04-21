#pragma once
#include <VolumeSlicer/CUDA/cuda_helper.hpp>
#include <VolumeSlicer/Data/slice.hpp>
VS_START

struct RawSampleParameter{
    uint32_t image_w;
    uint32_t image_h;
    float3 volume_board;//dim*space
    float3 space;
    float base_space;
    float2 voxels_per_pixel;
    float3 origin;
    float3 right;
    float3 down;
};

class CUDARawVolumeSampler{
public:
    CUDARawVolumeSampler(CUcontext ctx=nullptr)
    :old_h(0),old_w(0),cu_sample_result(nullptr),
    cu_volume_data(nullptr),volume_data_size(0),
    volume_x(0),volume_y(0),volume_z(0)
    {
        if(!ctx){
            CUDA_DRIVER_API_CALL(cuInit(0));
            int cu_device_cnt=0;
            CUdevice cu_device;
            int using_gpu=0;
            char using_device_name[80];
            CUDA_DRIVER_API_CALL(cuDeviceGetCount(&cu_device_cnt));
            CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device,using_gpu));
            CUDA_DRIVER_API_CALL(cuDeviceGetName(using_device_name,sizeof(using_device_name),cu_device));
            CUDA_DRIVER_API_CALL(cuCtxCreate(&cu_ctx,0,cu_device));
        }
        else{
            this->cu_ctx=ctx;
        }
    };
    ~CUDARawVolumeSampler();

    void SetCUDACtx(){
        CUDA_DRIVER_API_CALL(cuCtxSetCurrent(cu_ctx));
    }

    void SetVolumeData(uint8_t* data,uint32_t dim_x,uint32_t dim_y,uint32_t dim_z);

    void Sample(uint8_t* data,Slice slice,float space_x,float space_y,float space_z);
private:
    CUcontext cu_ctx;

    int old_w,old_h;
    uint8_t* cu_sample_result;//image


    cudaArray* cu_volume_data;
    cudaTextureObject_t volume_texture;

    size_t volume_data_size;
    uint32_t volume_x,volume_y,volume_z;
};

VS_END