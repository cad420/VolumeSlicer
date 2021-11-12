//
// Created by wyz on 2021/10/8.
//
#pragma once

#include <VolumeSlicer/Ext/iblock_volume_plugin_interface.hpp>
#include <VolumeSlicer/Ext/ih264_volume_plugin_interface.hpp>
#include <VolumeSlicer/Ext/plugin.hpp>
#include <VolumeSlicer/cuda_mem_pool.hpp>
#include <VolumeSlicer/utils.hpp>

VS_START

using VolumeBlock = typename CompVolume::VolumeBlock;

class Worker;

class BlockVolumeProviderPlugin : public IBlockVolumeProviderPluginInterface
{
  public:
    BlockVolumeProviderPlugin();

    ~BlockVolumeProviderPlugin() override;

    void Open(std::string const &filename) override;

    auto GetBlockDim(int lod) const -> std::array<uint32_t, 3> override;

    //{block_length,padding,min_lod,max_lod}
    auto GetBlockLength() const -> std::array<uint32_t, 4> override;

    // return true if the block index is valid and GetAvailableNum()>0
    // todo: bool or void
    bool AddTask(std::array<uint32_t, 4> const &) override;

    // get block from the internal container
    auto GetBlock() -> CompVolume::VolumeBlock override;

    // return if can get any valid block
    bool IsEmpty() override;

    // used for multi-thread implement, return empty thread num for load block
    size_t GetAvailableNum() override;

    // used for multi-thread implement, return if all threads are ready to load block
    bool IsAllAvailable() override;

  private:
    size_t block_size_bytes;

    int cu_mem_num;
    std::unique_ptr<CUDAMemoryPool<uint8_t>> cu_mem_pool;

    std::unique_ptr<IH264VolumeReaderPluginInterface> packet_reader;

    int worker_num;
    std::vector<Worker> workers;

    std::unique_ptr<ThreadPool> jobs;

    ConcurrentQueue<VolumeBlock> products;
};

VS_END

class BlockVolumeProviderPluginFactory : public vs::IPluginFactory
{
  public:
    std::string Key() const override
    {
        return "VolumeBlock";
    }
    void *Create(const std::string &key) override
    {
        if (key == "VolumeBlock")
        {
            return new vs::BlockVolumeProviderPlugin();
        }
        return nullptr;
    }
    std::string GetModuleID() const override
    {
        return {"VolumeSlicer.volume.loader"};
    }
};

VS_REGISTER_PLUGIN_FACTORY_DECL(BlockVolumeProviderPluginFactory)
EXPORT_PLUGIN_FACTORY_DECL(BlockVolumeProviderPluginFactory)