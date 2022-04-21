//
// Created by wyz on 2021/6/10.
//

#pragma once

#include <VolumeSlicer/CUDA/cuda_helper.hpp>

VS_START

template <typename T>
struct Alloc{

};

template<class T>
class CUDAMemAllocator{
public:
    CUDAMemAllocator()=default;

    static void alloc(T** p_device_ptr,size_t size){
        CUDA_DRIVER_API_CALL(cuMemAlloc((CUdeviceptr*)p_device_ptr,size));
    }

    static void free(T* device_ptr){
        CUDA_DRIVER_API_CALL(cuMemFree((CUdeviceptr)device_ptr));
    }
};


VS_END



