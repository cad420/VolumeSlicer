//
// Created by wyz on 2021/12/20.
//
#include <VolumeSlicer/memory_helper.hpp>
#include <VolumeSlicer/cuda_context.hpp>
#if defined(_WINDOWS) || defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else

#endif

VS_START

void MemoryHelper::GetGPUMemoryInfo(size_t &free, size_t &total)
{
    CUDA_DRIVER_API_CALL(cuMemGetInfo(&free,&total));
}

template <typename T>
void MemoryHelper::GetRecommendGPUTextureNum(int &num)
{
    size_t free,total;
    GetGPUMemoryInfo(free,total);
    free *= GPUMemoryUseRatio;
    free = (free>>30) > MAXGPUMemoryUsageGB ? (size_t(MAXGPUMemoryUsageGB)<<30) : free;
    num = free / sizeof(T) / DefaultGPUTextureSizeX / DefaultGPUTextureSizeY / DefaultGPUTextureSizeZ;
}

void MemoryHelper::GetCPUMemoryInfo(size_t &free, size_t &total)
{
#if defined(_WINDOWS) || defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    free = status.ullAvailPhys;
    total = status.ullTotalPhys;
#else

#endif
}

template <typename T>
void MemoryHelper::GetRecommendCPUTextureNum(int &num)
{
    size_t free,total;
    GetCPUMemoryInfo(free,total);
    free *= CPUMemoryUseRatio;
    free = (free>>30) > MAXCPUMemoryUsageGB ? (size_t(MAXCPUMemoryUsageGB)<<30) : free;
    num = free / sizeof(T) / DefaultCPUTextureSizeX / DefaultCPUTextureSizeY / DefaultCPUTextureSizeZ;
}

#define DefineGPUTextureType(T) EXPLICT_INSTANCE_TEMPLATE_FUNCTION(T,void,MemoryHelper::GetRecommendGPUTextureNum,int&)

DefineGPUTextureType(uint8_t)

DefineGPUTextureType(uint32_t)

#define DefineCPUTextureType(T) EXPLICT_INSTANCE_TEMPLATE_FUNCTION(T,void,MemoryHelper::GetRecommendCPUTextureNum,int&)

DefineCPUTextureType(uint8_t)

DefineCPUTextureType(uint32_t)

VS_END

