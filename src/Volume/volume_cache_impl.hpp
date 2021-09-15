//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_VOLUME_CACHE_IMPL_HPP
#define VOLUMESLICER_VOLUME_CACHE_IMPL_HPP

#include <VolumeSlicer/volume_cache.hpp>

VS_START

class CUDAVolumeBlockCacheImpl: public CUDAVolumeBlockCache{
public:
    explicit CUDAVolumeBlockCacheImpl(CUcontext ctx);

    ~CUDAVolumeBlockCacheImpl() override;

    void SetCacheBlockLength(uint32_t) override;

    void SetCacheCapacity(uint32_t num,uint32_t x,uint32_t y,uint32_t z) override;

    auto GetCacheShape()->std::array<uint32_t,4> override;

    void CreateMappingTable(const std::map<uint32_t,std::array<uint32_t,3>>&) override;

    void UploadVolumeBlock(const std::array<uint32_t,4>&,uint8_t*,size_t) override;

    //query if the block is cached
    bool IsCachedBlock(const std::array<uint32_t,4>&) override;

    bool IsValidBlock(const std::array<uint32_t,4>&) override;

    //std::array<bool,2>{valid,cached}
    auto GetBlockStatus(const std::array<uint32_t,4>&)->std::array<bool,2> override;

    //get number of block which is not valid, and don't care of whether is cached
    int  GetRemainEmptyBlock() const override;

    //set all blocks invalid
    void clear() override;

    //if target block is cached, then set it valid and return true and update mapping table
    //if target block is not cached, return false
    bool SetCachedBlockValid(const std::array<uint32_t,4>&) override;

    //just set target block invalid
    void SetBlockInvalid(const std::array<uint32_t,4>&) override;

    auto GetMappingTable()->const std::vector<uint32_t>& override;

    auto GetLodMappingTableOffset()->const std::map<uint32_t,uint32_t>& override;

    auto GetCUDATextureObjects()->std::vector<cudaTextureObject_t> override;

private:
    struct BlockCacheItem{
        std::array<uint32_t,4> block_index;
        std::array<uint32_t,4> pos_index;
        bool valid;
        bool cached;
    };
private:
    void createBlockCacheTable();
    void updateMappingTable(const std::array<uint32_t,4>& index,const std::array<uint32_t,4>& pos,bool valid=true);
    bool getCachedPos(const std::array<uint32_t,4>&,std::array<uint32_t,4>&);
private:

    uint32_t block_length;

    CUcontext cu_context;
    uint32_t cu_array_num;
    std::array<uint32_t,3> cu_array_shape;
    std::vector<cudaArray*> cu_arrays;
    std::vector<cudaTextureObject_t> cu_textures;

    std::list<BlockCacheItem> block_cache_table;

    std::vector<uint32_t> mapping_table;
    std::map<uint32_t,std::array<uint32_t,3>> lod_block_dim;
    std::map<uint32_t,uint32_t> lod_mapping_table_offset;
    uint32_t min_lod,max_lod;
};



VS_END

#endif //VOLUMESLICER_VOLUME_CACHE_IMPL_HPP
