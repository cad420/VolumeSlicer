//
// Created by wyz on 2021/6/7.
//

#include <fstream>
#include <chrono>

#include <VolumeSlicer/Ext/iblock_volume_plugin_interface.hpp>
#include <VolumeSlicer/Ext/iraw_volume_plugin_interface.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/plugin_loader.hpp>

#include "Data/block_loader.hpp"
#include "Data/volume_impl.hpp"

VS_START

/**************************************************************************************************
 * API for VolumeImpl<VolumeType::Raw>
 */

template <class Ty>
void LoadRawVolumeData(const char *file_name, std::vector<uint8_t> &volume_data)
{
    std::ifstream in(file_name, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("file open failed!");
    }
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<Ty> read_data;
    read_data.resize(file_size, 0);
    in.read(reinterpret_cast<char *>(read_data.data()), file_size);
    in.close();

    Ty min_value = std::numeric_limits<Ty>::max();
    Ty max_value = std::numeric_limits<Ty>::min();
    LOG_INFO("Type({0}) max value is {1}, min value is {2}.", typeid(Ty).name(), min_value, max_value);
    auto min_max = std::minmax_element(read_data.cbegin(), read_data.cend());
    min_value = *min_max.first;
    max_value = *min_max.second;
    LOG_INFO("Read volume data max value is {0}, min value is {1}.", max_value, min_value);
    volume_data.resize(file_size, 0);
    for (size_t i = 0; i < volume_data.size(); i++)
    {
        volume_data[i] = 1.f * (read_data[i] - min_value) / (max_value - min_value) * 255;
    }
}

std::unique_ptr<RawVolume> Volume<VolumeType::Raw>::Load(const char *file_name, VoxelType type,
                                                         const std::array<uint32_t, 3> &dim,
                                                         const std::array<float, 3> &space)
{
    try
    {
        auto f = std::unique_ptr<IRawVolumeReaderPluginInterface>(
            PluginLoader::CreatePlugin<IRawVolumeReaderPluginInterface>(".raw"));
        if (!f)
        {
            throw std::runtime_error("Plugin create failed.");
        }
        f->Open(file_name, type, dim);
        std::vector<uint8_t> volume_data;
        f->GetData(volume_data);

        std::unique_ptr<RawVolume> volume(new RawVolumeImpl(std::move(volume_data)));
        volume->SetDimX(dim[0]);
        volume->SetDimY(dim[1]);
        volume->SetDimZ(dim[2]);
        volume->SetSpaceX(space[0]);
        volume->SetSpaceY(space[1]);
        volume->SetSpaceZ(space[2]);
        return volume;
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("Plugin for raw volume read not found");
    }
    LOG_INFO("Using default method");

    std::vector<uint8_t> volume_data;
    switch (type)
    {
    case VoxelType::UInt8:
        LoadRawVolumeData<uint8_t>(file_name, volume_data);
        break;
    case VoxelType::UInt16:
        LoadRawVolumeData<uint16_t>(file_name, volume_data);
        break;
    case VoxelType::UInt32:
        LoadRawVolumeData<uint32_t>(file_name, volume_data);
        break;
    default:
        throw std::runtime_error("Unsupported voxel type");
    }
    std::unique_ptr<RawVolume> volume(new RawVolumeImpl(std::move(volume_data)));
    volume->SetDimX(dim[0]);
    volume->SetDimY(dim[1]);
    volume->SetDimZ(dim[2]);
    volume->SetSpaceX(space[0]);
    volume->SetSpaceY(space[1]);
    volume->SetSpaceZ(space[2]);
    return volume;

}

/**************************************************************************************************
 * API for VolumeImpl<VolumeType::Comp>
 */

std::unique_ptr<CompVolume> Volume<VolumeType::Comp>::Load(const char *file_name)
{
    return std::make_unique<CompVolumeImpl>(file_name);
}

CompVolumeImpl::VolumeImpl(const char *file_name) : pause(false), stop(false)
{
    try
    {
        this->block_loader = std::unique_ptr<IBlockVolumeProviderPluginInterface>(
            PluginLoader::CreatePlugin<IBlockVolumeProviderPluginInterface>("VolumeBlock"));
        if (!block_loader)
        {
            throw std::runtime_error("Plugin create failed.");
        }
        block_loader->Open(file_name);
        LOG_INFO("Create comp-volume loader plugin");
    }
    catch (const std::exception &err)
    {
        LOG_ERROR("{0}", err.what());
    }
    if(!block_loader)
    {
        try{
            LOG_INFO("Use default loader for comp-volume");
            this->block_loader = std::make_unique<BlockLoader>();
            this->block_loader->Open(file_name);
        }
        catch (const std::exception& err)
        {
            LOG_ERROR("{}",err.what());
            throw std::runtime_error("Failed to create block loader");
        }
    }
    this->block_queue.setSize(16);
    this->Loading();
    auto dim = block_loader->GetBlockDim(0);
    auto block_length = block_loader->GetBlockLength();
    this->n_voxels_x = dim[0] * (block_length[0] - block_length[1] * 2);
    this->n_voxels_y = dim[1] * (block_length[0] - block_length[1] * 2);
    this->n_voxels_z = dim[2] * (block_length[0] - block_length[1] * 2);
    auto space = block_loader->GetVolumeSpace();
    this->space_x = space[0];
    this->space_y = space[1];
    this->space_z = space[2];
}

void VolumeImpl<VolumeType::Comp>::ClearRequestBlock() noexcept
{
    std::lock_guard<std::mutex> lk(mtx);
    this->request_queue.clear();
}

void VolumeImpl<VolumeType::Comp>::SetRequestBlock(const std::array<uint32_t, 4> &idx) noexcept
{
    std::lock_guard<std::mutex> lk(mtx);
    if (!FindInRequestBlock(idx))
    {
        this->request_queue.push_back(idx);
    }
}

bool VolumeImpl<VolumeType::Comp>::FindInRequestBlock(const std::array<uint32_t, 4> &idx)
{
    //    std::unique_lock<std::mutex> lk(mtx);
    // std::any_of();
    for (auto &it : this->request_queue)
    {
        if (it == idx)
        {
            return true;
        }
    }
    return false;
}

void VolumeImpl<VolumeType::Comp>::EraseBlockInRequest(const std::array<uint32_t, 4> &idx) noexcept
{
    std::lock_guard<std::mutex> lk(mtx);
    for (auto it = this->request_queue.begin(); it != this->request_queue.end(); it++)
    {
        if (*it == idx)
        {
            this->request_queue.erase(it);
            break;
        }
    }
}

void VolumeImpl<VolumeType::Comp>::ClearBlockQueue() noexcept
{
    // clear blocks in block_queue which are not exist in request_queue
    std::lock_guard<std::mutex> lk(mtx);
    int queue_size = block_queue.size();
    while (queue_size-- > 0)
    {
        auto item = block_queue.pop_front();
        if (std::find(request_queue.begin(), request_queue.end(), item.index) == request_queue.end())
        {
            assert(item.valid && item.block_data);
            item.block_data->Release();
        }
        else
        {
            block_queue.push_back(item);
        }
    }
}

void VolumeImpl<VolumeType::Comp>::ClearBlockInQueue(const std::vector<std::array<uint32_t, 4>> &targets) noexcept
{
    // clear blocks in block_queue which are not exist in targets
    std::lock_guard<std::mutex> lk(mtx);
    int queue_size = block_queue.size();
    while (queue_size-- > 0)
    {
        auto item = block_queue.pop_front();
        if (std::find(targets.begin(), targets.end(), item.index) == targets.end())
        {
            assert(item.valid && item.block_data);
            item.block_data->Release();
            LOG_DEBUG("clear block in queue {0} {1} {2} {3}.", item.index[0], item.index[1], item.index[2], item.index[3]);
        }
        else
        {
            block_queue.push_back(item);
        }
    }
}

void VolumeImpl<VolumeType::Comp>::ClearAllBlockInQueue() noexcept
{
    std::lock_guard<std::mutex> lk(mtx);
    while (!block_queue.empty())
    {
        auto item = block_queue.pop_front();
        item.block_data->Release();
    }
}

int VolumeImpl<VolumeType::Comp>::GetBlockQueueSize()
{
    return block_queue.size();
}

Volume<VolumeType::Comp>::VolumeBlock VolumeImpl<VolumeType::Comp>::GetBlock(const std::array<uint32_t, 4> &idx) noexcept
{
    if (block_queue.find(idx))
    {
        return block_queue.get(idx);
    }
    else
    {
        return Volume<VolumeType::Comp>::VolumeBlock();
    }
}

VolumeBlock VolumeImpl<VolumeType::Comp>::GetBlock() noexcept
{
    if (block_queue.empty())
    {
        return Volume<VolumeType::Comp>::VolumeBlock();
    }
    else
    {
        auto block = block_queue.front();
        block_queue.pop_front();
        return block;
    }
}

void VolumeImpl<VolumeType::Comp>::Loading()
{
    task = std::thread([&]() {
        while (true)
        {
            if (this->stop)
            {
                LOG_DEBUG("stop and return.");
                return;
            }

            if (pause)
            {
                std::mutex _mtx;
                std::unique_lock<std::mutex> lk(_mtx);
                cv.wait(lk, [&]() {
                    if (pause)
                    {
                        paused = true;
                    }
                    return !pause;
                });
            }
            else
            {
                paused = false;
                // no blocking until AddBlocks()
                auto num = block_loader->GetAvailableNum();
                for (size_t i = 0; i < num; i++)
                {
                    auto req = FetchRequest();
                    block_loader->AddTask(req);
                }
                AddBlocks();
            }
        }
    });
}

void VolumeImpl<VolumeType::Comp>::StartLoadBlock() noexcept
{
    pause = false;
    cv.notify_all();
}

void VolumeImpl<VolumeType::Comp>::PauseLoadBlock() noexcept
{
    pause = true;

    while (!paused){}

    if (!paused)
    {
        LOG_DEBUG("not paused!!!!!!!");
    }
}

void VolumeImpl<VolumeType::Comp>::AddBlocks()
{
    std::lock_guard<std::mutex> lk(mtx);

    while (!block_loader->IsEmpty())
    {
        auto block = block_loader->GetBlock();
        if (!block.valid)
            continue;

        // assert get valid block if not empty but may get invalid in multi-thread
        assert(block.block_data->GetDataPtr());
        LOG_DEBUG("add to block_queue: {0} {1} {2} {3}.", block.index[0], block.index[1], block.index[2], block.index[3]);

        block_queue.push_back(block);
    }
}

auto VolumeImpl<VolumeType::Comp>::FetchRequest() -> std::array<uint32_t, 4>
{
    std::unique_lock<std::mutex> lk(mtx);
    if (request_queue.empty())
    {
        return {INVALID, INVALID, INVALID, INVALID};
    }
    else
    {
        auto req = request_queue.front();
        request_queue.pop_front();
        return req;
    }
}

auto VolumeImpl<VolumeType::Comp>::GetBlockDim(int lod) const -> std::array<uint32_t, 3>
{
    return block_loader->GetBlockDim(lod);
}

auto VolumeImpl<VolumeType::Comp>::GetBlockLength() const -> std::array<uint32_t, 4>
{
    return block_loader->GetBlockLength();
}

VolumeImpl<VolumeType::Comp>::~VolumeImpl()
{
    this->stop = true;
    this->pause = false;
    cv.notify_all();
    if (task.joinable())
        task.join();
    LOG_DEBUG("Delete comp_volume... Remain request num: {0}, block num: {1}.", request_queue.size(), block_queue.size());
    LOG_INFO("Exit CompVolume Loading...");
}

bool VolumeImpl<VolumeType::Comp>::GetStatus()
{
    std::lock_guard<std::mutex> lk(mtx);
    return request_queue.empty() && block_queue.empty() && block_loader->IsAllAvailable() && block_loader->IsEmpty();
}

void VolumeImpl<VolumeType::Comp>::SetBlockQueueSize(size_t size)
{
    this->block_queue.setSize(size);
}

auto VolumeImpl<VolumeType::Comp>::GetBlockDim() const -> const std::map<uint32_t, std::array<uint32_t, 3>> &
{
    static std::map<uint32_t, std::array<uint32_t, 3>> block_dim;
    uint32_t min_lod = GetBlockLength()[2];
    uint32_t max_lod = GetBlockLength()[3];
    for (auto i = min_lod; i <= max_lod; i++)
    {
        block_dim[i] = GetBlockDim(i);
    }
    return block_dim;
}

int VolumeImpl<VolumeType::Comp>::GetBlockQueueMaxSize()
{
    return block_queue.maxsize();
}

VS_END
