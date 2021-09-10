//
// Created by wyz on 2021/8/31.
//
#include <VolumeSlicer/render.hpp>
using namespace vs;
int main(){
    auto renderer=CPURawVolumeRenderer::Create(900,900);
    Camera camera;
    camera.zoom=45.0;
    camera.pos={1.28,1.28,5.0};
    camera.look_at={1.28,1.28,0.0};
    camera.up={0.0,1.0,0.0};
    camera.right={1.0,0.0,0.0};
    renderer->SetCamera(camera);
    auto volume=RawVolume ::Load("27#46#9#lod0_512_512_521_uint8.raw",VoxelType::UInt8,
                                   {512,512,512},
                                   {0.01,0.01,0.03});
    renderer->SetVolume(std::move(volume));
    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(25,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(60,std::array<double,4>{0.75,0.5,1.0,0.0});
    tf.points.emplace_back(64,std::array<double,4>{0.75,0.75,0.75,0.9});
    tf.points.emplace_back(224,std::array<double,4>{1.0,0.5,0.75,0.9});
    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
    renderer->SetTransferFunc(std::move(tf));

    renderer->render();

    auto image = renderer->GetImage();

    image.SaveToFile("cpu_raw_render_result.png");
}