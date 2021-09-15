//
// Created by wyz on 2021/9/14.
//
#include <VolumeSlicer/render.hpp>
#include <Utils/timer.hpp>
#include <iostream>
using namespace vs;
int main(){

    SetCUDACtx(0);
    auto renderer=CUDAOffScreenCompVolumeRenderer::Create(300,300);
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

    image.SaveToFile("cuda_comp_render_result.png");
    return 0;
}