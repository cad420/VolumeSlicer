//
// Created by wyz on 2021/6/10.
//

#pragma once

#include<VolumeSlicer/helper.hpp>

VS_START

template<class T>
class CUDAMemAllocator{
public:
    CUDAMemAllocator()=default;

    void alloc(T** p_device_ptr,size_t size){
        cuMemAlloc((CUdeviceptr*)p_device_ptr,size);
    }

    void free(T* device_ptr){
        cuMemFree((CUdeviceptr)device_ptr);
    }
};


VS_END



