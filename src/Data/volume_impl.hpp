//
// Created by wyz on 2021/6/7.
//
#pragma once

#include <vector>

#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Ext/iblock_volume_plugin_interface.hpp>
#include <VolumeSlicer/Utils/utils.hpp>

VS_START

class BlockLoader;

template <VolumeType type>
class VolumeImpl;

template <>
class VolumeImpl<VolumeType::Raw> : public Volume<VolumeType::Raw>
{
  public:
    VolumeImpl(std::vector<uint8_t> &&data) : raw_volume_data(std::move(data)){};

    ~VolumeImpl() noexcept override
    {
    }

    VolumeType GetVolumeType() const override
    {
        return VolumeType::Raw;
    }

    uint8_t *GetData() override
    {
        return raw_volume_data.data();
    };

  private:
    std::vector<uint8_t> raw_volume_data;
};
using RawVolumeImpl = VolumeImpl<VolumeType::Raw>;

template <>
class VolumeImpl<VolumeType::Comp> : public CompVolume
{
  public:
    explicit VolumeImpl(const char *file_name);

    // base class must define ~ function
    ~VolumeImpl();

    VolumeType GetVolumeType() const override
    {
        return VolumeType::Comp;
    }

    VolumeBlock GetBlock(const std::array<uint32_t, 4> &) noexcept override;

    VolumeBlock GetBlock() noexcept override;

    auto GetBlockDim(int lod) const -> std::array<uint32_t, 3> override;

    auto GetBlockDim() const -> const std::map<uint32_t, std::array<uint32_t, 3>> & override;

    auto GetBlockLength() const -> std::array<uint32_t, 4> override;

  protected:
    void ClearRequestBlock() noexcept override;

    void SetRequestBlock(const std::array<uint32_t, 4> &) noexcept override;

    void EraseBlockInRequest(const std::array<uint32_t, 4> &) noexcept override;

    void ClearBlockQueue() noexcept override;

    void ClearBlockInQueue(const std::vector<std::array<uint32_t, 4>> &targets) noexcept override;

    void ClearAllBlockInQueue() noexcept override;

    int GetBlockQueueSize() override;

    int GetBlockQueueMaxSize() override;

    void SetBlockQueueSize(size_t size) override;

    void PauseLoadBlock() noexcept override;

    void StartLoadBlock() noexcept override;

    bool GetStatus() override;

  private:
    bool FindInRequestBlock(const std::array<uint32_t, 4> &idx);

    /**
     * @brief pop and return a request from request_queue, if empty return INVALID request,
     * so user must check the returned value.
     * @note this function is non-blocking.
     */
    auto FetchRequest() -> std::array<uint32_t, 4>;

    /**
     * @brief fetch block from block_loader and push to block_queue.
     * @note this function will block if block_queue is full.
     */
    void AddBlocks();

    void Loading();

  private:
    // while clear block_queue, can't add block to the queue.
    // while operate on request_queue, loader can't operate
    std::mutex mtx;

    std::condition_variable cv;
    std::atomic<bool> pause;
    std::atomic<bool> paused;

    //it will async load block using cpu or gpu depend on the detail implement.
    //it works as a producer and also has a max-size storage for production.
    std::unique_ptr<IBlockVolumeProviderPluginInterface> block_loader;

    //thread for send decode block index to block_loader and fetch decoded block data from block_loader
    std::thread task;
    //if stop task will terminate
    bool stop;

    //block index for block_loader to load
    std::list<std::array<uint32_t, 4>> request_queue;

    //production queue that stores the decoded volume block,
    //it's thread-safe and has a max-size.
    //it will block when pop if empty and push if full.
    ConcurrentQueue<VolumeBlock> block_queue;
};
using CompVolumeImpl = VolumeImpl<VolumeType::Comp>;

VS_END
