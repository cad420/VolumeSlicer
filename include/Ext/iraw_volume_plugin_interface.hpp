//
// Created by wyz on 2021/9/30.
//
#pragma once
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/volume.hpp>
#include <Ext/plugin_define.hpp>
VS_START


class IRawVolumeReaderPluginInterface{
  public:
    virtual void Open(const std::string& filename,
                      VoxelType type,
                      std::array<uint32_t,3> const& dim) = 0;

    virtual void GetData(std::vector<uint8_t>& data) = 0;

    virtual void Close() = 0;
};

DECLARE_PLUGIN_MODULE_ID(IRawVolumeReaderPluginInterface,"VolumeSlicer.volume.reader")

VS_END



