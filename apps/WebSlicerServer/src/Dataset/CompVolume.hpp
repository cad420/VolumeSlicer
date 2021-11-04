//
// Created by wyz on 2021/11/2.
//
#pragma once
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/volume.hpp>
VS_START
namespace remote{

class VolumeDataSet{
  public:
    static void Load(const std::string&);
    static auto GetVolume()->const std::shared_ptr<CompVolume>&;
  private:
    static std::shared_ptr<CompVolume> comp_volume;
};

}
VS_END