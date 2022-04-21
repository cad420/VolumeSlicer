//
// Created by wyz on 2021/7/30.
//

#pragma once

#include <VolumeSlicer/Common/color.hpp>
#include <VolumeSlicer/Render/render.hpp>
#include <VolumeSlicer/Utils/linear_array.hpp>

#include <VolumeSlicer/Utils/sampler.hpp>
#include <VolumeSlicer/Utils/texture.hpp>

VS_START

class CPURawVolumeRendererImpl : public CPURawVolumeRenderer
{
  public:
    explicit CPURawVolumeRendererImpl(int w, int h);

    void SetMPIRender(MPIRenderParameter) override;

    auto GetBackendName() -> std::string override;

    void SetStep(double step, int steps) override;

    void SetVolume(std::shared_ptr<RawVolume> raw_volume) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunc(TransferFunc tf) override;

    void render(bool sync) override;

    auto GetImage() -> const Image<Color4b> & override;

    void resize(int w, int h) override;

    void clear() override;

  private:
    Image<Color4b> image; // image can easily save to file

    Texture1D<Color4f> tf_1d;
    Texture2D<Color4f> tf_2d;
    Texture3D<uint8_t> volume_data;
    Linear3DArray<uint8_t> volume_data_array;
    std::vector<uint32_t> cdf_map;
    static constexpr int cdf_block_length = 4;
    int cdf_dim_x, cdf_dim_y, cdf_dim_z;
    double volume_board_x, volume_board_y, volume_board_z;
    double space_x, space_y, space_z;
    int volume_dim_x, volume_dim_y, volume_dim_z;
    int window_w, window_h;
    Camera camera;
};

VS_END
