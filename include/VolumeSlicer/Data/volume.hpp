//
// Created by wyz on 2021/6/7.
//

#pragma once

#include <map>
#include <type_traits>

#include <VolumeSlicer/CUDA/cuda_memory.hpp>
#include <VolumeSlicer/Data/slice.hpp>

VS_START

enum class VS_EXPORT VoxelType{
    Int8,
    UInt8,
    Int16,
    UInt16,
    Float16,
    Int32,
    UInt32,
    Float32,
    Float64
};

enum class VS_EXPORT VolumeType{
    Raw,
    Comp
};

class VolumeBase {
public:
    virtual ~VolumeBase(){}

    virtual VolumeType GetVolumeType() const=0;

    void SetSpaceX(float space){this->space_x=space;}
    void SetSpaceY(float space){this->space_y=space;}
    void SetSpaceZ(float space){this->space_z=space;}

    void SetDimX(uint32_t x){this->n_voxels_x=x;}
    void SetDimY(uint32_t y){this->n_voxels_y=y;}
    void SetDimZ(uint32_t z){this->n_voxels_z=z;}

    auto GetVolumeDimX() const -> uint32_t {return n_voxels_x;}
    auto GetVolumeDimY() const -> uint32_t {return n_voxels_y;}
    auto GetVolumeDimZ() const -> uint32_t {return n_voxels_z;}

    auto GetVolumeSpaceX() const -> float {return space_x;}
    auto GetVolumeSpaceY() const -> float {return space_y;}
    auto GetVolumeSpaceZ() const -> float {return space_z;}
protected:
    uint32_t n_voxels_x = 0;
    uint32_t n_voxels_y = 0;
    uint32_t n_voxels_z = 0;
    float space_x = 1.f;
    float space_y = 1.f;
    float space_z = 1.f;
};

template<VolumeType type>
class Volume;

template<>
class VS_EXPORT Volume<VolumeType::Raw>: public VolumeBase,public std::true_type {
public:
    Volume() = default;

    virtual ~Volume() noexcept {}

    static std::unique_ptr<Volume<VolumeType::Raw>> Load(const char* file_name,VoxelType type,
                                                         const std::array<uint32_t,3>& dim,
                                                         const std::array<float,3>& space );

    //voxel will convert to uint8_t while Load using min-max stretch
    virtual uint8_t* GetData() = 0;
};

using RawVolume=Volume<VolumeType::Raw>;

template<>
class VS_EXPORT Volume<VolumeType::Comp>: public VolumeBase,public std::false_type{
public:
    struct alignas(16) VolumeBlock{

        //be careful to use
        VolumeBlock(const VolumeBlock& block){
            this->index = block.index;
            this->block_data = block.block_data;
            this->valid = block.valid;
        }

        VolumeBlock(VolumeBlock&& block){
            this->index = block.index;
            this->valid = block.valid;
            this->block_data = std::move(block.block_data);
            block.valid = false;
        }

        VolumeBlock():valid(false){}

        bool operator==(const std::array<uint32_t,4>& idx) const{
            return index == idx;
        }

        ~VolumeBlock(){
            block_data.reset();
        }

        //must explicitly call by user if not use this VolumeBlock any more
        //release the occupy for the CUDAMem
        void Release(){
            block_data->Release();
            valid = false;
        }

        std::array<uint32_t,4> index;//3+1: index + lod

        std::shared_ptr<CUDAMem<uint8_t>> block_data;//if not used any more, user should call Release

        bool valid;//false rep nullptr for block_data and invalid, should not be used
    };

public:
    Volume() = default;

    virtual ~Volume() noexcept {};

    /**
     * @brief create comp-volume object, if failed will throw exception
     * @param file_name compressed volume data config file path
     */
    static std::unique_ptr<Volume<VolumeType::Comp>> Load(const char* file_name);

    //will return immediately(no blocking)
    //VolumeBlock::valid is false meanings not get supposed block and cant's use
    //if returned VolumeBlock's data used, should call Release
    virtual VolumeBlock GetBlock(const std::array<uint32_t,4>&) noexcept = 0;

    //get block in the front of block queue, if queue is empty will return invalid block
    virtual VolumeBlock GetBlock() noexcept = 0;

    //get lod volume's dim: dim-xyz + padding
    virtual auto GetBlockDim(int lod) const ->std::array<uint32_t,3>  = 0;

    virtual auto GetBlockDim() const -> const std::map<uint32_t,std::array<uint32_t,3>>& = 0;

    //get comp-volume's (block_length, padding, min_lod, max_lod)
    virtual auto GetBlockLength() const ->std::array<uint32_t,4> = 0;

    REGISTER_FRIEND_CLASS(CLASS_IMPLEMENT(CUDACompVolumeRenderer))
    REGISTER_FRIEND_CLASS(CLASS_IMPLEMENT(OpenGLCompVolumeRenderer))
    REGISTER_FRIEND_CLASS(CLASS_IMPLEMENT(CPUOffScreenCompVolumeRenderer))
    REGISTER_FRIEND_CLASS(CLASS_IMPLEMENT(CUDAOffScreenCompVolumeRenderer))
    REGISTER_TEMPLATE_FRIEND_CLASS(CLASS_IMPLEMENT(VolumeSampler))
protected:

    //clear all blocks in request
    virtual void ClearRequestBlock() noexcept {}

    //set blocks in request
    virtual void SetRequestBlock(const std::array<uint32_t,4>&) noexcept {}

    //erase a certain block in request
    virtual void EraseBlockInRequest(const std::array<uint32_t,4>&) noexcept {}

    //clear blocks which are not in current request blocks
    virtual void ClearBlockQueue() noexcept {}

    //clear blocks in queue which are not in targets
    virtual void ClearBlockInQueue(const std::vector<std::array<uint32_t,4>>& targets) noexcept {}

    //clear all blocks in queue
    virtual void ClearAllBlockInQueue() noexcept {}

    //return block num in queue which can get
    virtual int GetBlockQueueSize() { return 0; }

    //return max block num in result queue
    virtual int GetBlockQueueMaxSize() { return 0; }

    //set maximum number of blocks which have uncompressed in the block queue
    virtual void SetBlockQueueSize(size_t) {}

    //pause loading meanings block queue will not increase
    virtual void PauseLoadBlock() noexcept {}

    //restart loading block in request
    virtual void StartLoadBlock() noexcept {}

    //return if request and block_queue are all empty
    virtual bool GetStatus() { return false; }
};

using CompVolume=Volume<VolumeType::Comp>;



VS_END

