//
// Created by wyz on 2021/10/8.
//
#pragma once

#include <VolumeSlicer/Ext/ih264_volume_plugin_interface.hpp>
#include <VolumeSlicer/Ext/plugin.hpp>
#include <VolumeSlicer/Utils/LRU.hpp>
#include <VolumeSlicer/Utils/hash.hpp>

#include <array>
#include <fstream>

#include <VoxelCompression/voxel_compress/VoxelCmpDS.h>

VS_START

class H264VolumeReaderPlugin : public IH264VolumeReaderPluginInterface
{
  public:
    H264VolumeReaderPlugin();

    ~H264VolumeReaderPlugin();

    void Open(std::string const &filename) override;

    void AddLodData(int lod, std::string const &filename) override;

    void GetPacket(std::array<uint32_t, 4> const &idx, std::vector<std::vector<uint8_t>> &packet) override;

    size_t GetBlockSizeByte() override;

    auto GetBlockLength() const -> std::array<uint32_t, 4> override;

    auto GetBlockDim(int lod) const -> std::array<uint32_t, 3> override;

    auto GetFrameShape() const -> std::array<uint32_t, 2> override;

    auto GetVolumeSpace() const -> std::array<float,3> override;

  private:
    std::unordered_map<int, std::unique_ptr<sv::Reader>> readers;
    int min_lod, max_lod;
    std::array<float,3> volume_space;
    LRUCache<std::array<uint32_t, 4>, std::vector<std::vector<uint8_t>>> packet_cache;
    std::mutex mtx;
};

VS_END

class H264VolumeReaderPluginFactory : public vs::IPluginFactory
{
  public:
    std::string Key() const override
    {
        return ".h264";
    }
    void *Create(std::string const &key) override
    {
        if (key == ".h264")
        {
            return new vs::H264VolumeReaderPlugin();
        }
        return nullptr;
    }
    std::string GetModuleID() const override
    {
        return "VolumeSlicer.volume.reader";
    }
};

VS_REGISTER_PLUGIN_FACTORY_DECL(H264VolumeReaderPluginFactory)
EXPORT_PLUGIN_FACTORY_DECL(H264VolumeReaderPluginFactory)
