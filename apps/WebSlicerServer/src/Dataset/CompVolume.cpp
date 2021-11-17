//
// Created by wyz on 2021/11/2.
//
#include "CompVolume.hpp"
VS_START
static std::string GetExt(const std::string& path){
    auto pos = path.find_last_of('.');
    return path.substr(pos);
}
namespace remote
{

std::shared_ptr<CompVolume> VolumeDataSet::comp_volume = nullptr;
std::shared_ptr<RawVolume> VolumeDataSet::raw_volume = nullptr;
void VolumeDataSet::Load(const std::string &path)
{
    auto ext = GetExt(path);
    if(ext == ".json"){
        loadCompVolume(path);
    }
    else if(ext == ".raw"){
        loadRawVolume(path);
    }
    else{
        LOG_ERROR("invalid file ext");
    }
}
auto VolumeDataSet::GetVolume()->const std::shared_ptr<CompVolume>&
{
    if(!comp_volume){
        throw std::runtime_error("GetVolume before it load!!!");
    }
    return comp_volume;
}
void VolumeDataSet::loadCompVolume(const std::string &path)
{
    comp_volume = CompVolume::Load(path.c_str());
    comp_volume->SetSpaceX(0.00032f);
    comp_volume->SetSpaceY(0.00032f);
    comp_volume->SetSpaceZ(0.001f);
}
void VolumeDataSet::loadRawVolume(const std::string &path)
{
    raw_volume = RawVolume::Load(path.c_str(),VoxelType::UInt8,
                                 {366,463,161},
                                 {0.00032f,0.00032f,0.001f}
                                 );
}
auto VolumeDataSet::GetRawVolume() -> const std::shared_ptr<RawVolume> &
{
    if(!raw_volume){
        throw std::runtime_error("GetVolume before it load!!!");
    }
    return raw_volume;
}
}
VS_END