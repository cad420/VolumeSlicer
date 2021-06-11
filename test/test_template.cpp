//
// Created by wyz on 2021/6/8.
//
#include<VolumeSlicer/volume.hpp>
#include<iostream>
#include<list>
using namespace vs;
int main()
{



    std::cout<<sizeof(std::shared_ptr<CUDAMem<uint8_t>>)<<std::endl;
    std::cout<<sizeof(Volume<VolumeType::Comp>::VolumeBlock)<<std::endl;
    auto volume=Volume<VolumeType::Raw>::Load("111",VoxelType::UInt8,{1,1,1},{1.f,1.f,1.f});


}
