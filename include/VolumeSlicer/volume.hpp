//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_VOLUME_HPP
#define VOLUMESLICER_VOLUME_HPP


#include<VolumeSlicer/slice.hpp>
#include<VolumeSlicer/cuda_memory.hpp>

VS_START

enum class VoxelType{
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

enum class VolumeType{
    Raw,
    Comp
};

class VolumeBase{
public:
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
    uint32_t n_voxels_x;
    uint32_t n_voxels_y;
    uint32_t n_voxels_z;
    float space_x;
    float space_y;
    float space_z;
};

template<VolumeType type>
class Volume;

template<>
class Volume<VolumeType::Raw>: public VolumeBase{
public:

    static std::unique_ptr<Volume<VolumeType::Raw>> Load(const char* file_name,VoxelType type,
                        const std::array<uint32_t,3>& dim,
                        const std::array<float,3>& space );

    //voxel will convert to uint8_t while Load
    virtual uint8_t* GetData()=0;
};



template<>
class Volume<VolumeType::Comp>: public VolumeBase{
public:
    struct alignas(16) VolumeBlock{
        std::array<uint32_t,4> index;
        std::shared_ptr<CUDAMem<uint8_t>> block_data;
        int lod;
    };
public:
    //comp file must be uint8_t
    static std::unique_ptr<Volume<VolumeType::Comp>> Load(const char* file_name);

    virtual void RequestBlock(const std::array<uint32_t,4>&) noexcept =0;
    virtual void EraseBlock(const std::array<uint32_t,4>&) noexcept =0;
    virtual int GetQueueSize() const=0;
    virtual void PauseLoadBlock() noexcept = 0;
    virtual void StartLoadBlock() noexcept = 0;
    virtual VolumeBlock GetBlock(const std::array<uint32_t,4>&) noexcept =0;
};



class VS_EXPORT VolumeSampler{
public:
    VolumeSampler()=default;


    static std::unique_ptr<VolumeSampler> CreateVolumeSampler(const std::shared_ptr<VolumeBase>&);

    //data should has be alloc and its size equal to slice.n_pixels_width * slice.n_pixels_height
    //data could be cuda device ptr or cpu host ptr
    virtual void Sample(const Slice& slice,uint8_t* data)=0;



};

VS_END


#endif //VOLUMESLICER_VOLUME_HPP
