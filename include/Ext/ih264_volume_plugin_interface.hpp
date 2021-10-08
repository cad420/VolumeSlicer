//
// Created by wyz on 2021/10/8.
//
#pragma once

#include <VolumeSlicer/export.hpp>
#include <Ext/plugin_define.hpp>
#include <string>
#include <vector>
VS_START

class IH264VolumeReaderPluginInterface{
  public:
    ~IH264VolumeReaderPluginInterface() = default;

    virtual void Open(std::string const& filename) = 0;

    virtual void AddLodData(int lod,std::string const& filename) = 0;

    virtual void GetPacket(std::array<uint32_t,4> const& idx,std::vector<std::vector<uint8_t>>& packet) = 0;

    virtual size_t GetBlockSizeByte() = 0;

    virtual auto GetBlockLength() const -> std::array<uint32_t,4> = 0;

    virtual auto GetBlockDim(int lod) const -> std::array<uint32_t,3> = 0;

    virtual auto GetFrameShape() const -> std::array<uint32_t,2> = 0;
};

DECLARE_PLUGIN_MODULE_ID(IH264VolumeReaderPluginInterface,"VolumeSlicer.volume.reader")

VS_END