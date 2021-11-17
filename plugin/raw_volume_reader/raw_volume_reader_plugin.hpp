//
// Created by wyz on 2021/9/30.
//
#pragma once
#include <VolumeSlicer/Ext/iraw_volume_plugin_interface.hpp>
#include <VolumeSlicer/Ext/plugin.hpp>

#include <fstream>

VS_START

class RawVolumeReaderPlugin : public IRawVolumeReaderPluginInterface
{
  public:
    RawVolumeReaderPlugin() = default;

    ~RawVolumeReaderPlugin();

    void Open(const std::string &filename, VoxelType type, std::array<uint32_t, 3> const &dim) override;

    void GetData(std::vector<uint8_t> &data) override;

    void Close() override;

  private:
    std::ifstream in;
    std::array<uint32_t, 3> dim;
    VoxelType voxel_type;
};

VS_END

class RawVolumeReaderPluginFactory : public vs::IPluginFactory
{
  public:
    std::string Key() const override
    {
        return ".raw";
    }
    void *Create(const std::string &key) override
    {
        if (key == ".raw")
        {
            return new vs::RawVolumeReaderPlugin();
        }
        return nullptr;
    }
    std::string GetModuleID() const override
    {
        return {"VolumeSlicer.volume.reader"};
    };
};

VS_REGISTER_PLUGIN_FACTORY_DECL(RawVolumeReaderPluginFactory)
EXPORT_PLUGIN_FACTORY_DECL(RawVolumeReaderPluginFactory)