//
// Created by wyz on 2021/8/31.
//
#include <VolumeSlicer/render.hpp>
#include <Utils/timer.hpp>
#include <Utils/plugin_loader.hpp>
using namespace vs;
int main(){
    auto renderer=CPURawVolumeRenderer::Create(900,900);
    Camera camera;
    camera.zoom=45.0;
    camera.pos={1.28,1.28,3.5};
    camera.look_at={1.28,1.28,0.64};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
    renderer->SetCamera(camera);
//    PluginLoader::LoadPlugins("./plugins");
    auto volume=RawVolume ::Load("../test_data/engine_256_256_128_uint8.raw",VoxelType::UInt8,
                                   {256,256,128},
                                   {0.01,0.01,0.01});
    renderer->SetVolume(std::move(volume));
    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(19,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(45,std::array<double,4>{0.0,0.0,0.0,0.121});
    tf.points.emplace_back(57,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(102,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(107,std::array<double,4>{1.0,1.0,1.0,0.0});
    tf.points.emplace_back(122,std::array<double,4>{1.0,1.0,1.0,0.0});
    tf.points.emplace_back(124,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(132,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(140,std::array<double,4>{0.17,0.17,0.17,0.0});
    tf.points.emplace_back(155,std::array<double,4>{0.17,0.17,0.17,0.0});
    tf.points.emplace_back(162,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(199,std::array<double,4>{1.0,1.0,1.0,0.0});
    tf.points.emplace_back(246,std::array<double,4>{0.22,0.678,0.98,0.211});
    tf.points.emplace_back(255,std::array<double,4>{0.22,0.678,0.98,0.211});
    renderer->SetTransferFunc(std::move(tf));

    {
        AutoTimer timer;
        renderer->render();
    }
    auto image = renderer->GetImage();

    image.SaveToFile("cpu_raw_render_result.png");
}