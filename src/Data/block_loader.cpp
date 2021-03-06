//
// Created by wyz on 2021/6/9.
//

#include <VolumeSlicer/Utils/logger.hpp>

#include "Data/block_loader.hpp"

#include <VoxelCompression/voxel_uncompress/VoxelUncompress.h>

VS_START

class Worker
{
  public:
    Worker(const VoxelUncompressOptions &opt)
    {
        uncmp = std::make_unique<VoxelUncompress>(opt);
        status._a = false;
    }
    bool isBusy() const
    {
        return status._a;
    }
    void setStatus(bool _status)
    {
        status._a = _status;
    }
    void uncompress(uint8_t *dest_ptr, int64_t len, std::vector<std::vector<uint8_t>> &packets)
    {
        uncmp->uncompress(dest_ptr, len, packets);
    }

  private:
    std::unique_ptr<VoxelUncompress> uncmp;
    atomic_wrapper<bool> status;
};

BlockLoader::BlockLoader() : block_size_bytes(0), cu_mem_num(16), worker_num(2)
{
}

void BlockLoader::Open(const std::string &filename)
{
    this->packet_reader = Reader::CreateReader(filename.c_str());
    //only after create reader then can know block's information
    this->block_size_bytes = packet_reader->GetBlockSizeByte();
    LOG_DEBUG("block_size_bytes is: {0}.", block_size_bytes);
    this->cu_mem_pool = std::make_unique<CUDAMemoryPool<uint8_t>>(cu_mem_num, block_size_bytes);

    VoxelUncompressOptions uncmp_opts;
    auto frame_shape = packet_reader->GetFrameShape();
    uncmp_opts.width = frame_shape[0];
    uncmp_opts.height = frame_shape[1];
    uncmp_opts.use_device_frame_buffer = true;
    uncmp_opts.cu_ctx = GetCUDACtx();
    for (int i = 0; i < worker_num; i++)
        workers.emplace_back(uncmp_opts);

    jobs = std::make_unique<ThreadPool>(worker_num);

    products.setSize(cu_mem_num * 2); // max is cu_mem_num
}

size_t BlockLoader::GetAvailableNum()
{
    size_t num = 0;
    for (auto &worker : workers)
    {
        if (!worker.isBusy())
        {
            num++;
        }
    }

    return num;
}

bool BlockLoader::AddTask(const std::array<uint32_t, 4> &idx)
{
    // check if idx is valid
    if (idx[0] == INVALID || idx[1] == INVALID || idx[2] == INVALID || idx[3] == INVALID)
    {
        return false;
    }

    if (GetAvailableNum() == 0)
    {
        return false;
    }
    else
    {
        for (size_t i = 0; i < workers.size(); i++)
        {
            if (!workers[i].isBusy())
            {
                workers[i].setStatus(true);
                LOG_DEBUG("worker {0} append task.", i);
                jobs->AppendTask(
                    [&](int worker_id, const std::array<uint32_t, 4> &idx) {
                        std::vector<std::vector<uint8_t>> packet;
                        packet_reader->GetPacket(idx, packet);
                        VolumeBlock block;
                        block.index = idx;
                        LOG_DEBUG("in AppendTask {0} {1} {2} {3}.", block.index[0], block.index[1], block.index[2], block.index[3]);

                        //if no available cuda memory will wait
                        block.block_data = cu_mem_pool->GetCUDAMem();

                        assert(block.block_data->GetDataPtr());
                        workers[worker_id].uncompress(block.block_data->GetDataPtr(), block_size_bytes, packet);

                        block.valid = true;
                        products.push_back(block);

                        workers[worker_id].setStatus(false);
                    },
                    i, idx);
                break;
            }
        }
        return true;
    }
}

bool BlockLoader::IsEmpty()
{
    return products.empty();
}

auto BlockLoader::GetBlock() -> Volume<VolumeType::Comp>::VolumeBlock
{
    if (IsEmpty())
    {
        VolumeBlock block;
        block.block_data = nullptr;
        block.valid = false;
        block.index = {INVALID, INVALID, INVALID, INVALID};
        return block;
    }
    else
    {
        LOG_DEBUG("before GetBlock, products size: {0}.", products.size());
        return products.pop_front();
    }
}

BlockLoader::~BlockLoader()
{
    //must destruct jobs first
    jobs.reset();
    workers.clear();
    products.clear();
    packet_reader.reset();
    cu_mem_pool.reset();
    LOG_INFO("Delete block_loader...Remain product num: {0}.", products.size());
}

auto BlockLoader::GetBlockDim(int lod) const -> std::array<uint32_t, 3>
{
    return packet_reader->GetBlockDim(lod);
}

auto BlockLoader::GetBlockLength() const -> std::array<uint32_t, 4>
{
    return packet_reader->GetBlockLength();
}

bool BlockLoader::IsAllAvailable()
{
    return GetAvailableNum() == worker_num;
}
auto BlockLoader::GetVolumeSpace() const -> std::array<float, 3>
{
    return packet_reader->GetVolumeSpace();
}

VS_END
