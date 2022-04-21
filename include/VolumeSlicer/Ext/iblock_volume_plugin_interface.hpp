//
// Created by wyz on 2021/10/6.
//

#pragma once

#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Ext/plugin_define.hpp>

VS_START

/**
 * @brief interface for loading blocks of every lod level
 */
class IBlockVolumeProviderPluginInterface
{
  public:
    virtual ~IBlockVolumeProviderPluginInterface() = default;

    virtual void Open(std::string const &filename) = 0;

    virtual auto GetBlockDim(int lod) const -> std::array<uint32_t, 3> = 0;

    // return {block_length,padding,min_lod,max_lod}
    virtual auto GetBlockLength() const -> std::array<uint32_t, 4> = 0;

    virtual auto GetVolumeSpace() const -> std::array<float,3> = 0;

    /**
     * @brief add a loading task and non-blocking.
     * @param block_index
     * @return true if the block index is valid and GetAvailableNum()>0.
     */
    virtual bool AddTask(std::array<uint32_t, 4> const & block_index) = 0;

    /**
     * @brief get a block which is loaded.
     * @return a valid VolumeBlock or an invalid one if empty
     */
    virtual auto GetBlock() -> CompVolume::VolumeBlock = 0;

    // return if can get any valid block
    virtual bool IsEmpty() = 0;

    // used for multi-thread implement, return empty thread num for load block
    virtual size_t GetAvailableNum() = 0;

    // used for multi-thread implement, return if all threads are ready to load block
    virtual bool IsAllAvailable() = 0;
};

DECLARE_PLUGIN_MODULE_ID(IBlockVolumeProviderPluginInterface, "VolumeSlicer.volume.loader")

VS_END
