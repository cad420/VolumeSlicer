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
    camera.look_at={1.28,1.28,0.0};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
    renderer->SetCamera(camera);
//    PluginLoader::LoadPlugins("./plugins");
    auto volume=RawVolume ::Load("../test_data/aneurism_256_256_256_uint8.raw",VoxelType::UInt8,
                                   {256,256,256},
                                   {0.01,0.01,0.01});
    renderer->SetVolume(std::move(volume));
    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(25,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(120,std::array<double,4>{0.75,0.5,1.0,0.0});
    tf.points.emplace_back(124,std::array<double,4>{0.75,0.75,0.75,0.9});
    tf.points.emplace_back(224,std::array<double,4>{1.0,0.5,0.75,0.9});
    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
    renderer->SetTransferFunc(std::move(tf));

    {
        AutoTimer timer;
        renderer->render();
    }
    auto image = renderer->GetImage();

    image.SaveToFile("cpu_raw_render_result.png");
}