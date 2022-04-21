//
// Created by wyz on 2021/6/10.
//

#pragma once

#include <mutex>

#include <VolumeSlicer/CUDA/alloc.hpp>
#include <VolumeSlicer/CUDA/cuda_context.hpp>
#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Common/status.hpp>

VS_START

template <class T, class Alloc = CUDAMemAllocator<T>>
class VS_EXPORT CUDAMem
{
  public:
    // size is byte size
    CUDAMem(size_t size) : device_ptr(nullptr), size(size), occupied(false)
    {

        this->cu_ctx = GetCUDACtx();

        Alloc::alloc(&device_ptr, size);
    }

    ~CUDAMem()
    {
        Destroy();
        LOG_DEBUG("Delete a CUDA memory.");
    }

    CUDAMem(const CUDAMem &) = delete;
    CUDAMem &operator=(const CUDAMem &) = delete;
    CUDAMem(CUDAMem &&) = delete;
    CUDAMem &operator=(CUDAMem &&) = delete;

    size_t GetSize() const
    {
        return size;
    }

    T *GetDataPtr()
    {
        return device_ptr;
    }

    bool IsValidPtr() const
    {
        return device_ptr;
    }

    bool IsOccupied() const
    {
        return occupied;
    }

    void SetOccupied()
    {
        std::lock_guard<std::mutex> lk(mtx);
        occupied = true;
    }

    void Destroy()
    {
        std::lock_guard<std::mutex> lk(mtx);
        Alloc::free(device_ptr);
        device_ptr = nullptr;
        size = 0;
        occupied = false;
    }

    virtual void Release()
    {
        std::lock_guard<std::mutex> lk(mtx);
        occupied = false;
    }

  protected:
    CUcontext cu_ctx;
    T *device_ptr;
    size_t size;
    bool occupied;
    std::mutex mtx;// a cuda memory may be accessed by multi-threads
};

VS_END
