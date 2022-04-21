//
// Created by wyz on 2021/6/10.
//

#pragma once

#include <VolumeSlicer/CUDA/cuda_memory.hpp>
#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <condition_variable>

#include <vector>

VS_START

template <class T>
class CUDAMemImpl : public CUDAMem<T>
{
  public:
    CUDAMemImpl(size_t size, std::condition_variable &cv) : CUDAMem<T>(size), cv(cv)
    {
    }

    void Release() override
    {
        std::unique_lock<std::mutex> lk(this->mtx);
        this->occupied = false;
        cv.notify_one();
    }

  private:
    std::condition_variable &cv;
};

template <class T> class VS_EXPORT CUDAMemoryPool
{
  public:
    // block_size is byte count for any type T
    explicit CUDAMemoryPool(size_t num, size_t block_size);

    ~CUDAMemoryPool();

    void Resize(size_t num);

    size_t GetTotalCUDAMemNum();

    size_t GetValidCUDAMemNum();

    std::shared_ptr<CUDAMem<T>> GetCUDAMem();

  private:
    size_t block_size;
    std::vector<std::shared_ptr<CUDAMem<T>>> cu_mems;
    std::mutex mtx;
    std::condition_variable cv;
};

template <class T>
CUDAMemoryPool<T>::CUDAMemoryPool(size_t num, size_t block_size) : block_size(block_size)
{
    cu_mems.reserve(num);
    for (size_t i = 0; i < num; i++)
    {
        cu_mems.emplace_back(new CUDAMemImpl<T>(block_size, cv));
    }
}

template <class T>
std::shared_ptr<CUDAMem<T>> CUDAMemoryPool<T>::GetCUDAMem()
{

    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() {
        for (auto &cu_mem : cu_mems)
        {
            if (!cu_mem->IsOccupied())
            {
                return true;
            }
        }
        return false;
    });

    for (auto &cu_mem : cu_mems)
    {
        if (!cu_mem->IsOccupied())
        {
            cu_mem->SetOccupied();
            return cu_mem;
        }
    }
}

template <class T>
size_t CUDAMemoryPool<T>::GetTotalCUDAMemNum()
{
    return cu_mems.size();
}

template <class T>
size_t CUDAMemoryPool<T>::GetValidCUDAMemNum()
{
    std::unique_lock<std::mutex> lk(mtx);
    size_t num = 0;
    for (auto &cu_mem : cu_mems)
    {
        if (!cu_mem->IsOccupied())
        {
            num++;
        }
    }
    return num;
}

template <class T>
void CUDAMemoryPool<T>::Resize(size_t num)
{
    if (num <= cu_mems.size())
    {
        cu_mems.resize(num);
    }
    else
    {
        for (size_t i = 0; i < num - cu_mems.size(); i++)
        {
            cu_mems.emplace_back(new CUDAMemImpl<T>(block_size, cv));
        }
    }
}

template <class T>
CUDAMemoryPool<T>::~CUDAMemoryPool()
{
    std::unique_lock<std::mutex> lk(mtx);
    for (auto &cu_mem : cu_mems)
    {
        cu_mem->Release();
        cu_mem->Destroy();
    }
    LOG_INFO("Delete CUDA memory pool.");
}

VS_END
