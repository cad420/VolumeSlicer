//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_VOLUME_CACHE_HPP
#define VOLUMESLICER_VOLUME_CACHE_HPP
#include <VolumeSlicer/define.hpp>
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/cuda_memory.hpp>
#include <Utils/block_array.hpp>
#include <cstdint>
#include <map>
#include <array>
#include <vector>
VS_START

/**
 * store blocks using mapping table
 */
class VS_EXPORT VolumeBlockCache{
public:
    VolumeBlockCache() = default;

    virtual ~VolumeBlockCache() = default;

    virtual void SetCacheBlockLength(uint32_t) = 0;

    virtual void SetCacheCapacity(uint32_t num,uint32_t x,uint32_t y,uint32_t z) = 0;

    virtual auto GetCacheShape()->std::array<uint32_t,4> = 0;

    virtual void CreateMappingTable(const std::map<uint32_t,std::array<uint32_t,3>>&) = 0;

    virtual void UploadVolumeBlock(const std::array<uint32_t,4>&,uint8_t*,size_t) = 0;

    //query if the block is cached
    virtual bool IsCachedBlock(const std::array<uint32_t,4>&) = 0;

    virtual bool IsValidBlock(const std::array<uint32_t,4>&) = 0;

    //std::array<bool,2>{valid,cached}
    virtual auto GetBlockStatus(const std::array<uint32_t,4>&)->std::array<bool,2> = 0;

    //get number of block which is not valid, and don't care of whether is cached
    virtual int  GetRemainEmptyBlock() const = 0;

    //set all blocks invalid
    virtual void clear() = 0;

    //if target block is cached, then set it valid and return true and update mapping table
    //if target block is not cached, return false
    virtual bool SetCachedBlockValid(const std::array<uint32_t,4>&) = 0;

    //just set target block invalid
    virtual void SetBlockInvalid(const std::array<uint32_t,4>&) = 0;

    virtual auto GetMappingTable()->const std::vector<uint32_t>& = 0;

    virtual auto GetLodMappingTableOffset()->const std::map<uint32_t,uint32_t>& = 0;
};

class VS_EXPORT CUDAVolumeBlockCache: public VolumeBlockCache{
public:
    CUDAVolumeBlockCache()=default;

    static std::unique_ptr<CUDAVolumeBlockCache> Create(CUcontext ctx=nullptr);

    virtual auto GetCUDATextureObjects()->std::vector<cudaTextureObject_t> = 0;

};

class VS_EXPORT OpenGLVolumeBlockCache: public VolumeBlockCache{
public:

};

template <typename Block3DArray>
class VS_EXPORT CPUVolumeBlockCache: public VolumeBlockCache{
public:
    static std::unique_ptr<CPUVolumeBlockCache> Create(const std::shared_ptr<Block3DArray>&);


};

VS_END



#endif //VOLUMESLICER_VOLUME_CACHE_HPP
