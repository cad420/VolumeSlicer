//
// Created by wyz on 2021/6/10.
//

#ifndef VOLUMESLICER_CUDA_MEMORY_HPP
#define VOLUMESLICER_CUDA_MEMORY_HPP

#include<mutex>

#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>
#include<VolumeSlicer/alloc.hpp>

VS_START

template<class T,class Alloc=CUDAMemAllocator<T>>
class VS_EXPORT CUDAMem{
public:
    //size is byte size
    CUDAMem(size_t size):alloc(Alloc()),device_ptr(nullptr),size(size),occupied(false){
        CUDA_DRIVER_API_CALL(cuInit(0));
        int cu_device_cnt=0;
        CUdevice cu_device;
        int using_gpu=0;
        char using_device_name[80];
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&cu_device_cnt));
        CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device,using_gpu));
        CUDA_DRIVER_API_CALL(cuDeviceGetName(using_device_name,sizeof(using_device_name),cu_device));
        this->cu_ctx=nullptr;
        CUDA_DRIVER_API_CALL(cuCtxCreate(&cu_ctx,0,cu_device));

        alloc.alloc(&device_ptr,size);
    }
    ~CUDAMem(){
        Destroy();
        spdlog::info("Delete a CUDA memory.");
    }
    CUDAMem(const CUDAMem&)=delete;
    CUDAMem& operator=(const CUDAMem&)=delete;
    CUDAMem(CUDAMem&&)=delete;
    CUDAMem& operator=(CUDAMem&&)=delete;

    size_t GetSize() const{return size;}
    T* GetDataPtr() {return device_ptr;}
    bool IsValidPtr() const{return device_ptr;}
    bool IsOccupied(){
        std::unique_lock<std::mutex> lk(mtx);
        return occupied;
    }
    void SetOccupied(){
        std::unique_lock<std::mutex> lk(mtx);
        occupied=true;
    }
    void Destroy(){
        std::unique_lock<std::mutex> lk(mtx);
        alloc.free(device_ptr);
        device_ptr= nullptr;
        size=0;
    }
    virtual void Release() {
        std::unique_lock<std::mutex> lk(mtx);
        occupied=false;
    }

protected:
    CUcontext cu_ctx;
    Alloc alloc;
    T* device_ptr;
    size_t size;
    bool occupied;
    std::mutex mtx;
};



VS_END

#endif //VOLUMESLICER_CUDA_MEMORY_HPP
