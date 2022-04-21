//
// Created by wyz on 2021/7/21.
//

#pragma once

#include <VolumeSlicer/CUDA/cuda_memory.hpp>
#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Data/memory_cache.hpp>
#include <VolumeSlicer/Utils/block_array.hpp>

#include <cstdint>
#include <map>
#include <array>
#include <vector>

VS_START

/**
 * @brief VolumeBlockCache is the physical storage that renderer will use.
 * It will handle all the volume data related things like upload the volume block and update mapping table.
 * @note It only support uint8 voxel.
 */
class VS_EXPORT VolumeBlockCache{
public:
    VolumeBlockCache() = default;

    virtual ~VolumeBlockCache() = default;

    virtual void SetCacheBlockLength(uint32_t) = 0;

    /**
     * @brief create num of physical texture with the specified xyz
     * @param num number of physical texture
     * @param x width of texture
     * @param y height of texture
     * @param z depth of texture
     */
    virtual void SetCacheCapacity(uint32_t num,uint32_t x,uint32_t y,uint32_t z) = 0;

    /**
     * @return {num,x,y,z}
     */
    virtual auto GetCacheShape()->std::array<uint32_t,4> = 0;

    virtual void CreateMappingTable(const std::map<uint32_t,std::array<uint32_t,3>>&) = 0;

    virtual void UploadVolumeBlock(const std::array<uint32_t,4>&,uint8_t*,size_t,bool device) = 0;

    //query if the block is cached
    virtual bool IsCachedBlock(const std::array<uint32_t,4>&) = 0;

    virtual bool IsValidBlock(const std::array<uint32_t,4>&) = 0;

    //std::array<bool,2>{valid,cached}
    virtual auto GetBlockStatus(const std::array<uint32_t,4>&)->std::array<bool,2> = 0;

    //get number of block which is not valid, and don't care of whether is cached
    virtual int  GetRemainEmptyBlock() const = 0;

    //may be different for derived class
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

    //get cuda implement texture handle
    virtual auto GetCUDATextureObjects()->std::vector<cudaTextureObject_t> = 0;

};

class VS_EXPORT OpenGLVolumeBlockCache: public VolumeBlockCache{
public:
    OpenGLVolumeBlockCache() =default;

    static std::unique_ptr<OpenGLVolumeBlockCache> Create();

    //get opengl implement texture handle
    virtual auto GetOpenGLTextureHandles() -> std::vector<uint32_t> = 0;
};

/**
 * @brief CPU volume block cache can be designed more simply not inherit from VolumeBlockCache.
 * @sa CPU has a better volume block cache see BlockCacheManager
 */
template <typename Block3DArray>
class VS_EXPORT [[deprecated]]  CPUVolumeBlockCache: public VolumeBlockCache{
public:
    [[deprecated]] static std::unique_ptr<CPUVolumeBlockCache> Create(const std::shared_ptr<Block3DArray>&);
};

VS_END



