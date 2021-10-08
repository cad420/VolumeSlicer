//
// Created by wyz on 2021/10/6.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/volume.hpp>
#include <Ext/plugin_define.hpp>
VS_START

/**
 * interface for load blocks of every lod level
 */
class IBlockVolumeProviderPluginInterface{
  public:
    virtual ~IBlockVolumeProviderPluginInterface()=default;

    virtual void Open(std::string const& filename) = 0;

    virtual auto GetBlockDim(int lod) const -> std::array<uint32_t,3> = 0;

    //{block_length,padding,min_lod,max_lod}
    virtual auto GetBlockLength() const -> std::array<uint32_t,4> = 0;

    //return true if the block index is valid and GetAvailableNum()>0
    //todo: bool or void
    virtual bool AddTask(std::array<uint32_t,4> const&) = 0;

    //get block from the internal container
    virtual auto GetBlock()->CompVolume::VolumeBlock = 0;

    //return if can get any valid block
    virtual bool IsEmpty() = 0;

    //used for multi-thread implement, return empty thread num for load block
    virtual size_t GetAvailableNum() = 0;

    //used for multi-thread implement, return if all threads are ready to load block
    virtual bool IsAllAvailable() = 0;
};

DECLARE_PLUGIN_MODULE_ID(IBlockVolumeProviderPluginInterface,"VolumeSlicer.volume.loader")

VS_END
