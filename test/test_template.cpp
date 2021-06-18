//
// Created by wyz on 2021/6/8.
//
#include<VolumeSlicer/volume.hpp>
#include<VolumeSlicer/render.hpp>
#include<VolumeSlicer/volume_sampler.hpp>
#include<iostream>
#include<list>
using namespace vs;
int main()
{
    std::shared_ptr<RawVolume> raw_volume=RawVolume::Load("C:\\Users\\wyz\\projects\\VolumeSlicer\\test_data\\aneurism_256_256_256_uint8.raw",VoxelType::UInt8,{256,256,256},{0.01f,0.01f,0.01f});

    Slice slice;
    slice.origin={128.f,128.f,128.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,1.f,-1.f,0.f};
    slice.normal={0.f,1.f,1.f,0.f};
    slice.n_pixels_height=400;
    slice.n_pixels_width=300;
    slice.voxel_per_pixel_height=1.f;
    slice.voxel_per_pixel_width=1.f;

    auto slicer=Slicer::CreateSlicer(slice);

    auto volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);


}
