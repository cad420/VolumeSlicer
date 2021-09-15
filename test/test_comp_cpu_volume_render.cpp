//
// Created by wyz on 2021/9/7.
//
#include <VolumeSlicer/render.hpp>
#include <Utils/timer.hpp>
#include <iostream>
#include <fstream>
using namespace vs;
int main(){

    SetCUDACtx(0);
    auto renderer=CPUOffScreenCompVolumeRenderer ::Create(300,300);
    Camera camera;
    //22.5 -> 450
    // 5    -> 50
    //turn 18
    camera.zoom=8;
    camera.pos={4.5f,7.5f,6.f};
    camera.look_at={4.5f,7.5f,0.0};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
    renderer->SetCamera(camera);
    auto volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
//    {
//        volume->SetRequestBlock({25,47,8,0});
//        _sleep(3000);
//        auto block=volume->GetBlock({25,47,8,0});
//
//        std::vector<uint8_t> v(512*512*512);
//        cudaMemcpy(v.data(),block.block_data->GetDataPtr(),v.size(),cudaMemcpyDefault);
//        std::fstream out("25#47#8#lod0_512_512_512_uint8.raw",std::ios::binary|std::ios::out);
//        if(!out.is_open()){
//            std::cout<<"not open"<<std::endl;
//            return 1;
//        }
//        out.write(reinterpret_cast<char*>(v.data()),v.size());
//        out.close();
//        return 0;
//    }

    volume->SetSpaceX(0.00032);
    volume->SetSpaceY(0.00032);
    volume->SetSpaceZ(0.001);
    renderer->SetVolume(std::move(volume));
    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.1,0.0,0.0,0.0});
    tf.points.emplace_back(25,std::array<double,4>{0.1,0.0,0.0,0.0});
    tf.points.emplace_back(30,std::array<double,4>{1.0,0.75,0.7,0.9});
    tf.points.emplace_back(64,std::array<double,4>{1.0,0.75,0.7,0.9});
    tf.points.emplace_back(224,std::array<double,4>{1.0,0.85,0.5,0.9});
    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,0.8,1.0});
    renderer->SetTransferFunc(std::move(tf));
    AutoTimer timer;
    renderer->render();

    auto image = renderer->GetImage();

    image.SaveToFile("cpu_comp_render_result.png");
    return 0;
}