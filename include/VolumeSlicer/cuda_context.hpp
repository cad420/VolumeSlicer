//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_CONTEXT_HPP
#define VOLUMESLICER_CUDA_CONTEXT_HPP

#include <VolumeSlicer/helper.hpp>

class CUDACtx{
public:
    CUDACtx(CUdevice d){
        CUDA_DRIVER_API_CALL(cuInit(0));
        int cu_device_cnt=0;
        CUdevice cu_device;
        char using_device_name[80];
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&cu_device_cnt));
        CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device,d));
        CUDA_DRIVER_API_CALL(cuDeviceGetName(using_device_name,sizeof(using_device_name),cu_device));
        CUDA_DRIVER_API_CALL(cuCtxCreate(&cu_ctx,0,cu_device));
        spdlog::info("CUDA device count: {0}, using device {1}: {2}.",cu_device_cnt,cu_device,using_device_name);
    }
    CUcontext GetCUDACtx() const{
        return cu_ctx;
    }
private:
    CUcontext  cu_ctx;
};

VS_EXPORT void SetCUDACtx(CUdevice d);

VS_EXPORT CUcontext GetCUDACtx();

VS_EXPORT size_t GetCUDAFreeMem();

VS_EXPORT size_t GetCUDAUsedMem();

#endif //VOLUMESLICER_CUDA_CONTEXT_HPP
