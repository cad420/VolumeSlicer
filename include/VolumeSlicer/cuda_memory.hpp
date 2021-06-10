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
        alloc.alloc(&device_ptr,size);
    }
    ~CUDAMem(){
        alloc.free(device_ptr);
    }
    CUDAMem(const CUDAMem&)=delete;
    CUDAMem& operator=(const CUDAMem&)=delete;
    CUDAMem(CUDAMem&&)=delete;
    CUDAMem& operator=(CUDAMem&&)=delete;

    size_t GetSize() const{return size;}
    T* GetDataPtr() {return device_ptr;}
    bool IsValidPtr() const{return device_ptr;}
    bool IsOccupied() const{
        std::unique_lock<std::mutex> lk(mtx);
        return occupied;
    }
    void SetOccupied(){
        std::unique_lock<std::mutex> lk(mtx);
        occupied=true;
    }
    virtual void Release() {
        std::unique_lock<std::mutex> lk(mtx);
        occupied=false;
    }

protected:
    Alloc alloc;
    T* device_ptr;
    size_t size;
    bool occupied;
    std::mutex mtx;
};



VS_END

#endif //VOLUMESLICER_CUDA_MEMORY_HPP
