//
// Created by wyz on 2021/10/20.
//
#pragma once
#include <VolumeSlicer/volume_cache.hpp>
VS_START

class OpenGLVolumeBlockCacheImpl: public OpenGLVolumeBlockCache{
  public:
    explicit OpenGLVolumeBlockCacheImpl();

    ~OpenGLVolumeBlockCacheImpl() override;

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

    auto GetOpenGLTextureHandles() -> std::vector<uint32_t> override;
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
    std::list<BlockCacheItem> block_cache_table;

    std::vector<uint32_t> mapping_table;
    std::map<uint32_t,std::array<uint32_t,3>> lod_block_dim;
    std::map<uint32_t,uint32_t> lod_mapping_table_offset;
    uint32_t min_lod,max_lod;


    uint32_t block_length;

    uint32_t gl_tex_num;
    std::array<uint32_t,3> gl_tex_shape;
    std::vector<uint32_t> gl_textures;

    std::vector<CUgraphicsResource> cu_resources;
};

VS_END