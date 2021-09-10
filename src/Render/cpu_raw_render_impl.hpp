//
// Created by wyz on 2021/7/30.
//

#ifndef VOLUMESLICER_CPU_RAW_RENDER_IMPL_HPP
#define VOLUMESLICER_CPU_RAW_RENDER_IMPL_HPP

#include "Texture/sampler.hpp"
#include "Texture/texture.hpp"
#include <VolumeSlicer/color.hpp>
#include <VolumeSlicer/render.hpp>
VS_START

class CPURawVolumeRendererImpl : public CPURawVolumeRenderer
{
  public:
    explicit CPURawVolumeRendererImpl(int w, int h);

    void SetMPIRender(MPIRenderParameter) override;

    void SetStep(double step,int steps) override;

    void SetVolume(std::shared_ptr<RawVolume> raw_volume) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunc(TransferFunc tf) override;

    void render() override;

    auto GetFrame() -> const Image<uint32_t> & override;

    auto GetImage()->const Image<Color4b>& override;

    void resize(int w, int h) override;

    void clear() override;

  private:
    Image<uint32_t> frame;//frame suit for display
    Image<Color4b > image;//image can easily save to file

    Texture1D<Color4f> tf_1d;
    Texture2D<Color4f> tf_2d;
    Texture3D<uint8_t> volume_data;
    double volume_board_x,volume_board_y,volume_board_z;
    double space_x,space_y,space_z;
    int volume_dim_x,volume_dim_y,volume_dim_z;
    int window_w,window_h;
    Camera camera;
};

VS_END

#endif // VOLUMESLICER_CPU_RAW_RENDER_IMPL_HPP
