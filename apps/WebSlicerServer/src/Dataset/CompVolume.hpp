//
// Created by wyz on 2021/11/2.
//
#pragma once
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Data/volume.hpp>
VS_START
namespace remote{

class VolumeDataSet{
  public:
    static void Load(const std::string&);
    static auto GetVolume()->const std::shared_ptr<CompVolume>&;
    static auto GetRawVolume() ->const std::shared_ptr<RawVolume>&;
  private:
    static void loadCompVolume(const std::string&);
    static void loadRawVolume(const std::string&);
  private:
    static std::shared_ptr<CompVolume> comp_volume;
    static std::shared_ptr<RawVolume> raw_volume;
};

}
VS_END