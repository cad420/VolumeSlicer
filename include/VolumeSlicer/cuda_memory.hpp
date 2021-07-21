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
#include<VolumeSlicer/cuda_context.hpp>
VS_START

template<class T,class Alloc=CUDAMemAllocator<T>>
class VS_EXPORT CUDAMem{
public:
    //size is byte size
    CUDAMem(size_t size):alloc(Alloc()),device_ptr(nullptr),size(size),occupied(false){

        this->cu_ctx=GetCUDACtx();

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
