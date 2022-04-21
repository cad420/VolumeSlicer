//
// Created by wyz on 2021/6/9.
//

#pragma once

#include <vector>

#include "Data/volume_impl.hpp"

#include <VolumeSlicer/CUDA/cuda_mem_pool.hpp>
#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/IO/reader.hpp>

VS_START

using VolumeBlock = typename VolumeImpl<VolumeType::Comp>::VolumeBlock;
class Worker;

class BlockLoader : public IBlockVolumeProviderPluginInterface
{
  public:
    explicit BlockLoader();

    ~BlockLoader() override;

    void Open(std::string const &filename) override;

    // num of not busy decode worker
    size_t GetAvailableNum() override;

    // will check first in function, and only add valid block
    bool AddTask(const std::array<uint32_t, 4> &) override;

    // if there have decoded block data
    bool IsEmpty() override;

    auto GetBlock() -> CompVolume::VolumeBlock override;

    bool IsAllAvailable() override;

  public:
    auto GetBlockDim(int lod) const -> std::array<uint32_t, 3> override;

    auto GetBlockLength() const -> std::array<uint32_t, 4> override;

    auto GetVolumeSpace() const -> std::array<float,3> override;

  private:
    size_t block_size_bytes;

    int cu_mem_num;
    std::unique_ptr<CUDAMemoryPool<uint8_t>> cu_mem_pool;

    std::unique_ptr<Reader> packet_reader;

    int worker_num;
    std::vector<Worker> workers;

    std::unique_ptr<ThreadPool> jobs;

  public:
    ConcurrentQueue<VolumeBlock> products;
};

VS_END
