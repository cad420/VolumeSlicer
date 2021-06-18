#pragma once
#include<VolumeSlicer/helper.hpp>
#include<VolumeSlicer/slice.hpp>
#include<glm/glm.hpp>
VS_START

struct SampleParameter{
    uint32_t image_w;
    uint32_t image_h;
    float3 volume_board;//dim*space
    float2 voxels_per_pixel;
    float3 origin;
    float3 right;
    float3 down;
};

class CUDARawVolumeSampler{
public:
    CUDARawVolumeSampler()
    :old_h(0),old_w(0),cu_sample_result(nullptr),
    cu_volume_data(nullptr),volume_data_size(0),
    volume_x(0),volume_y(0),volume_z(0)
    {

    };

    void SetVolumeData(uint8_t* data,uint32_t dim_x,uint32_t dim_y,uint32_t dim_z);

    void sample(uint8_t* data,Slice slice,float space_x,float space_y,float space_z);
private:
    int old_w,old_h;
    uint8_t* cu_sample_result;//image


    cudaArray* cu_volume_data;
    cudaTextureObject_t volume_texture;

    size_t volume_data_size;
    uint32_t volume_x,volume_y,volume_z;
};

VS_END