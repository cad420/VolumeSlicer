//
// Created by wyz on 2021/6/9.
//

#ifndef VOLUMESLICER_BLOCK_LOADER_HPP
#define VOLUMESLICER_BLOCK_LOADER_HPP

#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/define.hpp>
#include<VolumeSlicer/reader.hpp>
#include<Common/cuda_mem_pool.hpp>
#include"Volume/volume_impl.hpp"
#include<vector>
VS_START

using VolumeBlock=typename VolumeImpl<VolumeType::Comp>::VolumeBlock;
class Worker;

class BlockLoader{
public:
    explicit BlockLoader(const char* file_path);
    ~BlockLoader();
    //num of not busy decode worker
    size_t GetAvailableNum();
    //will check first in function, and only add valid block
    void AddTask(const std::array<uint32_t,4>& );
    //if there have decoded block data
    bool IsEmpty();

    auto GetBlock()->CompVolume::VolumeBlock;
public:
    auto GetBlockDim(int lod) const ->std::array<uint32_t ,3> ;

    auto GetBlockLength() const ->std::array<uint32_t,4> ;
private:
    size_t block_size_bytes;

    int cu_mem_num;
    std::unique_ptr<CUDAMemoryPool<uint8_t>> cu_mem_pool;

    std::unique_ptr<Reader> packet_reader;

    int worker_num;
    std::vector<Worker> workers;

    std::unique_ptr<ThreadPool> jobs;

    ConcurrentQueue<VolumeBlock> products;
};

VS_END

#endif //VOLUMESLICER_BLOCK_LOADER_HPP
