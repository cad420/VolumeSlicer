//
// Created by wyz on 2021/9/28.
//
#include <VolumeSlicer/render.hpp>
#include "VideoCapture.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace vs;
int main(int argc,char** argv){
    SetCUDACtx(0);
    auto renderer=CUDAOffScreenCompVolumeRenderer::Create(1200,1200);
    Camera camera;
    camera.zoom=16;
    camera.pos={5.5f,6.5f,6.5f};
    camera.look_at={5.5f,6.5f,0.0};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
//    renderer->SetCamera(camera);
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

    CompRenderPolicy policy;
    policy.lod_dist[0]=0.3;
    policy.lod_dist[1]=0.5;
    policy.lod_dist[2]=1.2;
    policy.lod_dist[3]=1.6;
    policy.lod_dist[4]=3.2;
    policy.lod_dist[5]=6.4;
    policy.lod_dist[6]=std::numeric_limits<double>::max();
    renderer->SetRenderPolicy(policy);

    {
        VideoCapture video_capture("test_cuda_comp_render_video.avi", 1200, 1200, 30);

        for (int i = 0; i < 300; i++)
        {
            std::cout << "render frame " << i << std::endl;
            renderer->SetCamera(camera);
            renderer->render();
            auto image = renderer->GetImage().ToImage3b();
            video_capture.AddFrame(reinterpret_cast<uint8_t *>(image.GetData()));
            camera.pos[2] -= 0.01f;
        }
    }

    return 0;
}