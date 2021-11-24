//
// Created by wyz on 2021/9/14.
//
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/Utils/timer.hpp>
#include <iostream>
using namespace vs;
int main(){

    SetCUDACtx(0);
    auto renderer=CUDAOffScreenCompVolumeRenderer::Create(1920,1080);
    Camera camera;
    //22.5 -> 450
    // 5    -> 50
    //turn 18
    camera.zoom=16;
    camera.pos={5.5f,5.5f,5.98f};
    camera.look_at={5.5f,5.5f,0.0};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
    renderer->SetCamera(camera);

    auto volume=CompVolume::Load("D:/MouseNeuronData/mouse_file_config.json");
    volume->SetSpaceX(0.00032);
    volume->SetSpaceY(0.00032);
    volume->SetSpaceZ(0.001);
    renderer->SetVolume(std::move(volume));

    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
//    tf.points.emplace_back(25,std::array<double,4>{0.1,0.0,0.0,0.0});
//    tf.points.emplace_back(30,std::array<double,4>{1.0,0.75,0.7,0.0});
    tf.points.emplace_back(74,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(75,std::array<double,4>{0.6,0.6,0.6,0.6});
//    tf.points.emplace_back(64,std::array<double,4>{1.0,0.85,0.75,0.6});
//    tf.points.emplace_back(224,std::array<double,4>{1.0,0.85,0.85,0.9});
    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
    renderer->SetTransferFunc(std::move(tf));

    CompRenderPolicy policy;
    policy.lod_dist[0]=0.3;
    policy.lod_dist[1]=0.5;
    policy.lod_dist[2]=1.2;
    policy.lod_dist[3]=1.6;
    policy.lod_dist[4]=3.2;
    policy.lod_dist[5]=6.4;
    policy.lod_dist[6]=std::numeric_limits<double>::max();
    renderer->SetRenderPolicy(policy);

    AutoTimer timer;
    renderer->render();

    auto image = renderer->GetImage();
//    "lod_policy": [1.6,2.4,3.2,4.8,6.4,8.0,-1.0],
    image.SaveToFile("cuda_comp_render_result.png");
    return 0;
}