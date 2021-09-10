//
// Created by wyz on 2021/9/1.
//

#pragma once
#include <Utils/block_array.hpp>
#include <Utils/hash.hpp>
#include <Utils/logger.hpp>
#include <vector>
#include <queue>
#include <VolumeSlicer/volume.hpp>
#include <VolumeSlicer/LRU.hpp>

VS_START

/**
 * Using Block3DArray as container
 * Because malloc large continuous memory may failed or performance badly when access,
 * so using multi Block3DArray for storage and this may easily extend to memory for GPU
 */


template<typename Block3DArray>
class BlockCacheManager{
  public:
    class PhysicalMemoryBlockIndex{


      public:
        using internal_type=uint32_t;
        PhysicalMemoryBlockIndex(internal_type x,internal_type y,internal_type z,internal_type w)
            :x(x),y(y),z(z),w(w)
        {}
        PhysicalMemoryBlockIndex()
            :x(INVALID),y(INVALID),z(INVALID),w(INVALID)
        {}
        internal_type X() const{
            return x;
        }
        internal_type Y() const{
            return y;
        }
        internal_type Z() const{
            return z;
        }
        internal_type Index() const{
            return w & 0xffff;
        }
        void SetValid(bool valid){
            w = ((internal_type)valid)<<16 | (w & 0xffff);
        }
        bool IsValid() const{
            if(w!=INVALID &&  (w & 0xffff0000)>>16)
                return true;
            else
                return false;
        }
        size_t Hash() const{
            return (size_t)(Index())<<48 | (size_t)(Z())<<32 | (size_t)(Y())<<16 | (size_t)(X());
        }
        bool Smaller(PhysicalMemoryBlockIndex const& idx) const{
            return Hash() < idx.Hash();
        }
        bool operator<(PhysicalMemoryBlockIndex const& idx) const{
            if(IsValid() && !idx.IsValid()){
                return true;
            }
            if(!IsValid() && idx.IsValid()){
                return false;
            }
            return Hash()<idx.Hash();
        }
      private:
        internal_type x,y,z,w;
    };
    using SizeType = size_t ;
    using Self     = BlockCacheManager<Block3DArray>;
    using DataType = typename Block3DArray::DataType;
    using VirtualBlockIndex = glm::ivec4;//std::array<uint32_t,4>;
    using LRUCacheTable = LRUCache<VirtualBlockIndex,PhysicalMemoryBlockIndex>;
    using IndexType =typename PhysicalMemoryBlockIndex::internal_type ;
    static constexpr SizeType block_length = Block3DArray::BlockLength();
  private:
    int num_array;
    std::vector<std::shared_ptr<Block3DArray>> arrays;
    std::unique_ptr<LRUCache<VirtualBlockIndex,PhysicalMemoryBlockIndex>> block_cache_table;
    std::priority_queue<PhysicalMemoryBlockIndex> remain_physical_blocks;
    std::mutex mtx;
  private:
    void InitPhysicalBlocks(){
        while(!remain_physical_blocks.empty())
            remain_physical_blocks.pop();
        for(IndexType i=0;i<arrays.size();i++){
            for(IndexType x=0;x<arrays[i]->BlockNumX();x++){
                for(IndexType y=0;y<arrays[i]->BlockNumY();y++){
                    for(IndexType z=0;z<arrays[i]->BlockNumZ();z++){
                        remain_physical_blocks.push({x,y,z,i});
                    }
                }
            }
        }
    }
  public:
    SizeType GetRemainPhysicalBlockNum() const{
        return remain_physical_blocks.size();
    }
    void InitManagerResource(){
        while(!remain_physical_blocks.empty())
            remain_physical_blocks.pop();
        InitPhysicalBlocks();
    }

    explicit BlockCacheManager(int n_arrays,int len_x,int len_y,int len_z)
    {
        for(int i=0;i<n_arrays;i++){
            arrays.emplace_back(std::make_shared<Block3DArray>(len_x,len_y,len_z,nullptr));
        }
        block_cache_table=std::make_unique<LRUCacheTable>(GetPhysicalMemoryBlockNum());
        InitPhysicalBlocks();
        LOG_INFO("BlockCacheManager create successfully.");
        LOG_INFO("array num: {0}, shape: {1} {2} {3}",arrays.size(),arrays[0]->BlockNumX(),
                 arrays[0]->BlockNumY(),arrays[1]->BlockNumZ());
        LOG_INFO("block_cache_table capacity: {0}, current size: {1}",block_cache_table->get_capacity(),
                 block_cache_table->get_size());
        LOG_INFO("remain_physical_blocks num: {0}",remain_physical_blocks.size());
    }



    BlockCacheManager(const Self&)=delete;
    BlockCacheManager& operator=(const Self&)=delete;


    SizeType GetPhysicalMemoryBlockNum() const{
        if(arrays.empty()) return 0;
        return arrays.size()*arrays[0]->BlockNum();
    }

    bool UploadBlockData(CompVolume::VolumeBlock block){
        VirtualBlockIndex virtual_block_index={block.index[0],block.index[1],block.index[2],block.index[3]};
        if(block_cache_table->exist_key(virtual_block_index)){
            block_cache_table->get_value_ptr(virtual_block_index);
        }
        else{
            if(remain_physical_blocks.empty()){
                if(block_cache_table->get_size()==0){
                    throw std::runtime_error("error: block_cache_table size is zero but remain_physical_blocks is empty.");
                }
                auto used_physical_index=block_cache_table->get_back().second;
                block_cache_table->pop_back();
                remain_physical_blocks.push(used_physical_index);
            }
            auto physical_index=remain_physical_blocks.top();
            remain_physical_blocks.pop();
            physical_index.SetValid(true);
            block_cache_table->emplace_back(virtual_block_index,physical_index);
            arrays[physical_index.Index()]->SetBlockData(physical_index.X(),physical_index.Y(),physical_index.Z(),
                                                         block.block_data->GetDataPtr());
        }
        block.Release();
        return true;
    }

    bool UploadBlockData(const VirtualBlockIndex& virtual_block_index,const DataType* data){
        if(block_cache_table->exist_key(virtual_block_index)){

        }
        else{
            if(remain_physical_blocks.empty()){
                if(block_cache_table->get_size()==0){
                    throw std::runtime_error("error: block_cache_table size is zero but remain_physical_blocks is empty.");
                }
                auto used_physical_index=block_cache_table->get_back().second;
                block_cache_table->pop_back();
                remain_physical_blocks.push(used_physical_index);
            }
            auto physical_index=remain_physical_blocks.top();
            remain_physical_blocks.pop();
            physical_index.SetValid(true);
            block_cache_table->emplace_back(virtual_block_index,physical_index);
            arrays[physical_index.Index()]->SetBlockData(physical_index.X(),physical_index.Y(),physical_index.Z(),data);

        }
    }

    //this class is for cpu render, so reuse of cached block is meaningless
    //so just store if a block is cached
    //block cached policy use LRU
    bool IsBlockDataCached(const VirtualBlockIndex& virtual_block_index) const{
        if(block_cache_table->exist_key(virtual_block_index)){
            return true;
        }
        else{
            return false;
        }
    }

    //if the block is cached will return the actual physical pos in the Block3DArray
    //else will return {INVALID,INVALID,INVALID,INVALID}
    PhysicalMemoryBlockIndex GetPhysicalBlockIndex(const VirtualBlockIndex& virtual_block_index){
        if(block_cache_table->exist_key(virtual_block_index)){
//            std::lock_guard<std::mutex> lk(this->mtx);
            return block_cache_table->get_value_without_move(virtual_block_index);
        }
        else{
            return {INVALID,INVALID,INVALID,INVALID};
        }
    }


    auto GetBlock3DArray(int index)->std::shared_ptr<Block3DArray>{
        return arrays[index];
    }



};


VS_END
