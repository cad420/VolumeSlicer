//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_VOLUME_HPP
#define VOLUMESLICER_VOLUME_HPP

#include<VolumeSlicer/slice.hpp>

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
    uint32_t n_voxels_x;
    uint32_t n_voxels_y;
    uint32_t n_voxels_z;
    float space_x;
    float space_y;
    float space_z;
};

template<VolumeType type>
class Volume{
public:
    Volume()=delete;
};

template<>
class Volume<VolumeType::Raw>: public VolumeBase{
public:
    template<VoxelType type>
    static void Load(const char* file_name);

    virtual VoxelType GetVoxelType() const=0;

    virtual void SetSpaceX(float)=0;
    virtual void SetSpaceY(float)=0;
    virtual void SetSpaceZ(float)=0;

    virtual uint8_t* GetData()=0;
};

template<>
class Volume<VolumeType::Comp>: public VolumeBase{
public:
    struct alignas(16) VolumeBlock{
        std::array<uint32_t,4> index;
        uint8_t* data;
        int lod;
        bool valid;
    };
public:
    //comp file must be uint8_t
    static void Load(const char* file_name);

    virtual VoxelType GetVoxelType() const=0;

    virtual void SetSpaceX(float)=0;
    virtual void SetSpaceY(float)=0;
    virtual void SetSpaceZ(float)=0;

    virtual VolumeBlock GetBlock(const std::array<uint32_t,4>&)=0;
};

class VS_EXPORT VolumeSampler{
public:
    VolumeSampler()=default;

    template<VolumeType type>
    static VolumeSampler* CreateVolumeSampler(const Volume<type>&);

    //data should has be alloc and its size equal to slice.n_pixels_width * slice.n_pixels_height
    //data could be cuda device ptr or cpu host ptr
    virtual void Sample(const Slice& slice,uint8_t* data)=0;



};

VS_END


#endif //VOLUMESLICER_VOLUME_HPP
