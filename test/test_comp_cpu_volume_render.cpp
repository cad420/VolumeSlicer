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
    camera.zoom=16;
    camera.pos={5.5f,6.5f,9.5f};
    camera.look_at={5.5f,6.5f,0.0};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
    renderer->SetCamera(camera);
    auto volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
    {
//     46 42 7 0; 46 40 2 0; 46 42 9 0;46 41 9 ,0
//        volume->SetRequestBlock({46,40,9,0});
//        _sleep(3000);
//        auto block=volume->GetBlock({46,40,9,0});
//
//        std::vector<uint8_t> v(512*512*512);
//        cudaMemcpy(v.data(),block.block_data->GetDataPtr(),v.size(),cudaMemcpyDefault);
//        std::fstream out("46#40#9#lod01_512_512_512_uint8.raw",std::ios::binary|std::ios::out);
//        if(!out.is_open()){
//            std::cout<<"not open"<<std::endl;
//            return 1;
//        }
//        out.write(reinterpret_cast<char*>(v.data()),v.size());
//        out.close();
//        return 0;
    }

    volume->SetSpaceX(0.00032);
    volume->SetSpaceY(0.00032);
    volume->SetSpaceZ(0.001);
    renderer->SetVolume(std::move(volume));
    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.1,0.0,0.0,0.0});
    tf.points.emplace_back(25,std::array<double,4>{0.1,0.0,0.0,0.0});
    tf.points.emplace_back(30,std::array<double,4>{1.0,0.75,0.7,0.0});
    tf.points.emplace_back(60,std::array<double,4>{1.0,0.75,0.7,0.0});
    tf.points.emplace_back(64,std::array<double,4>{1.0,0.85,0.75,0.6});
    tf.points.emplace_back(224,std::array<double,4>{1.0,0.85,0.85,0.9});
    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,0.9,1.0});
    renderer->SetTransferFunc(std::move(tf));

    CompRenderPolicy policy;
    policy.lod_dist[0]=0.3;
    policy.lod_dist[1]=0.5;
    policy.lod_dist[2]=1.2;
    policy.lod_dist[3]=1.6;
    policy.lod_dist[4]=3.2;
    policy.lod_dist[5]=6.4;
    policy.lod_dist[6]=std::numeric_limits<double>::max();
    policy.cdf_value_file="chebyshev_dist_mouse_cdf_config.json";
//    policy.volume_value_file="volume_value_mouse_cdf_config.json";
    renderer->SetRenderPolicy(policy);

    AutoTimer timer;
    renderer->render();

    auto image = renderer->GetImage();

    image.SaveToFile("cpu_comp_render_result.png");
    return 0;
}