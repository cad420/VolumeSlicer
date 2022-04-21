//
// Created by wyz on 2021/10/6.
//
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Utils/plugin_loader.hpp>
using namespace vs;
void test_raw_volume(){
    PluginLoader::LoadPlugins("./plugins");
    auto raw_volume=RawVolume::Load("../test_data/aneurism_256_256_256_uint8.raw",
                                    VoxelType::UInt8,
                                    {256,256,256},
                                    {0.01f,0.01f,0.01f});
}
void test_comp_volume(){
    PluginLoader::LoadPlugins("./plugins");
    SetCUDACtx(0);
    auto comp_volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
}
int main(){
    test_comp_volume();

    return 0;
}