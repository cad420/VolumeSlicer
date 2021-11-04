//
// Created by wyz on 2021/11/2.
//
#include "CompVolume.hpp"
VS_START
namespace remote
{
std::shared_ptr<CompVolume> VolumeDataSet::comp_volume = nullptr;
void VolumeDataSet::Load(const std::string &path)
{
    comp_volume = CompVolume::Load(path.c_str());
}
auto VolumeDataSet::GetVolume()->const std::shared_ptr<CompVolume>&
{
    if(!comp_volume){
        throw std::runtime_error("GetVolume before it load!!!");
    }
    return comp_volume;
}
}
VS_END