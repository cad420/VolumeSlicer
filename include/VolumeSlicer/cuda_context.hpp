//
// Created by wyz on 2021/7/21.
//

#pragma once

#include <VolumeSlicer/Utils/cuda_helper.hpp>
#include <cuda.h>
class CUDACtx
{
  public:
    CUDACtx(CUdevice d)
    {
        CUDA_DRIVER_API_CALL(cuInit(0));
        int cu_device_cnt = 0;
        char using_device_name[80];
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&cu_device_cnt));
        CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device, d));
        CUDA_DRIVER_API_CALL(cuDeviceGetName(using_device_name, sizeof(using_device_name), cu_device));
        CUDA_DRIVER_API_CALL(cuCtxCreate(&cu_ctx, 0, cu_device));
        LOG_INFO("CUDA device count: {0}, using device {1}: {2}.", cu_device_cnt, cu_device, using_device_name);
    }
    CUcontext GetCUDACtx() const
    {
        return cu_ctx;
    }
    CUdevice GetCUDADev() const{
        return cu_device;
    }
  private:
    CUcontext cu_ctx;
    CUdevice cu_device;
};

VS_EXPORT void SetCUDACtx(CUdevice d);

VS_EXPORT CUcontext GetCUDACtx();

VS_EXPORT CUdevice GetCUDADev();

VS_EXPORT size_t GetCUDAFreeMem();

VS_EXPORT size_t GetCUDAUsedMem();
