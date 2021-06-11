//
// Created by wyz on 2021/6/9.
//

#ifndef VOLUMESLICER_BLOCK_LOADER_HPP
#define VOLUMESLICER_BLOCK_LOADER_HPP

#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/define.hpp>
#include<Common/cuda_mem_pool.hpp>
#include<IO/reader_impl.hpp>
#include"Volume/volume_impl.hpp"

VS_START
using VolumeBlock=typename VolumeImpl<VolumeType::Comp>::VolumeBlock;
class Worker;
class BlockLoader{
public:
    BlockLoader(const char* file_path);

    size_t GetAvailableNum();
    void AddTask(const std::array<uint32_t,4>& );
    bool IsEmpty();
    auto GetBlock()->Volume<VolumeType::Comp>::VolumeBlock;
private:
    size_t block_size_bytes;

    std::unique_ptr<CUDAMemoryPool<uint8_t>> cu_mem_pool;
    std::unique_ptr<Reader> packet_reader;
    std::vector<Worker> workers;
    std::unique_ptr<ThreadPool> jobs;
    ConcurrentQueue<VolumeBlock> products;
};

VS_END

#endif //VOLUMESLICER_BLOCK_LOADER_HPP
