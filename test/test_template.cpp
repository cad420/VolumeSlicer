//
// Created by wyz on 2021/6/8.
//
#include<VolumeSlicer/volume.hpp>
#include<VolumeSlicer/render.hpp>
#include<VolumeSlicer/volume_sampler.hpp>
#include<VolumeSlicer/frame.hpp>
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
    slice.n_pixels_width=1200;
    slice.n_pixels_height=900;
    slice.voxel_per_pixel_height=1.f;
    slice.voxel_per_pixel_width=1.f;

    auto slicer=Slicer::CreateSlicer(slice);

    auto volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);

    std::shared_ptr<CompVolume> comp_volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
    auto block_length=comp_volume->GetBlockLength();
    std::cout<<"block length: "<<block_length[0]<<" "<<block_length[1]<<std::endl;
    auto block_dim=comp_volume->GetBlockDim(0);
    std::cout<<"block dim: "<<block_dim[0]<<" "<<block_dim[1]<<" "<<block_dim[2]<<std::endl;
//    comp_volume->PauseLoadBlock();
//    std::cout<<"set 0"<<std::endl;
//    comp_volume->SetRequestBlock({0,0,0,0});
//    std::cout<<"set 1"<<std::endl;
//    comp_volume->SetRequestBlock({0,0,0,1});
//    std::cout<<"set 2"<<std::endl;
//    comp_volume->SetRequestBlock({0,0,0,2});
//    std::cout<<"set 3"<<std::endl;
//    comp_volume->SetRequestBlock({0,0,0,3});
//    comp_volume->StartLoadBlock();
//    while(true){
//        _sleep(2000);
//        break;
//    };
    auto comp_volume_sampler=VolumeSampler::CreateVolumeSampler(comp_volume);
    //todo
    comp_volume->SetSpaceX(0.01f);
    comp_volume->SetSpaceY(0.01f);
    comp_volume->SetSpaceZ(0.01f);
    Frame frame;
    frame.width=slicer->GetImageW();
    frame.height=slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height
    *frame.channels,0);
    comp_volume_sampler->Sample(slicer->GetSlice(),frame.data.data());
}
