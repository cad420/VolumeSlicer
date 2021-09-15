//
// Created by wyz on 2021/9/1.
//

#pragma once
#include <VolumeSlicer/volume_cache.hpp>
VS_START

template <typename Block3DArray>
std::unique_ptr<CPUVolumeBlockCache<Block3DArray>> CPUVolumeBlockCache<Block3DArray>::Create(const std::shared_ptr<Block3DArray> &block_array)
{
    return std::make_unique<CPUVolumeBlockCacheImpl<Block3DArray>>(block_array);
}

template <typename Block3DArray>
class CPUVolumeBlockCacheImpl: public CPUVolumeBlockCache<Block3DArray>{
  public:
    using Self     = CPUVolumeBlockCacheImpl<Block3DArray>;
    using SizeType = size_t ;
    CPUVolumeBlockCacheImpl(const std::shared_ptr<Block3DArray>& block_array){

    }

    void SetCacheBlockLength(uint32_t) override{

    }

    void SetCacheCapacity(uint32_t num,uint32_t x,uint32_t y,uint32_t z) override{

    }

    auto GetCacheShape()->std::array<uint32_t,4> override{
        return {};
    }

    void CreateMappingTable(const std::map<uint32_t,std::array<uint32_t,3>>&) override{

    }

    void UploadVolumeBlock(const std::array<uint32_t,4>&,uint8_t*,size_t) override{

    }

    //query if the block is cached
    bool IsCachedBlock(const std::array<uint32_t,4>&) override{
        return false;
    }

    bool IsValidBlock(const std::array<uint32_t,4>&) override{
        return false;
    }

    //std::array<bool,2>{valid,cached}
    auto GetBlockStatus(const std::array<uint32_t,4>&)->std::array<bool,2> override{
        return {};
    }
    //get number of block which is not valid, and don't care of whether is cached
    int  GetRemainEmptyBlock() const {
        return 0;
    }

    //set all blocks invalid
    void clear(){

    }

    //if target block is cached, then set it valid and return true and update mapping table
    //if target block is not cached, return false
    bool SetCachedBlockValid(const std::array<uint32_t,4>&) override{
        return false;
    }

    //just set target block invalid
    void SetBlockInvalid(const std::array<uint32_t,4>&) override{

    }

    auto GetMappingTable()->const std::vector<uint32_t>& override{
        return {};
    }

    auto GetLodMappingTableOffset()->const std::map<uint32_t,uint32_t>& override{
        return {};
    }

};

#define EXPLICT_INSTANCE_TEMPLATE_CLASS(CLS,T) \
    template class CLS<T>;

#define EXPLICT_INSTANCE_TEMPLATE_TEMPLATE_CLASS(CLS,T,...) \
    template class CLS<T<__VA_ARGS__>>;

EXPLICT_INSTANCE_TEMPLATE_TEMPLATE_CLASS(CPUVolumeBlockCache,Block3DArray,uint8_t,8)
EXPLICT_INSTANCE_TEMPLATE_TEMPLATE_CLASS(CPUVolumeBlockCacheImpl,Block3DArray,uint8_t,8)

EXPLICT_INSTANCE_TEMPLATE_TEMPLATE_CLASS(CPUVolumeBlockCache,Block3DArray,uint8_t,9)
EXPLICT_INSTANCE_TEMPLATE_TEMPLATE_CLASS(CPUVolumeBlockCacheImpl,Block3DArray,uint8_t,9)

VS_END
