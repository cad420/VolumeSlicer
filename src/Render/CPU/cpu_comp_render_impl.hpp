//
// Created by wyz on 2021/7/30.
//

#pragma once

#include <VolumeSlicer/Common/vec.hpp>
#include <VolumeSlicer/Data/cdf.hpp>
#include <VolumeSlicer/Render/render.hpp>
#include <VolumeSlicer/Utils/block_cache.hpp>

#include <VolumeSlicer/Utils/sampler.hpp>

VS_START

class CPUOffScreenCompVolumeRendererImpl : public CPUOffScreenCompVolumeRenderer
{
  public:
    CPUOffScreenCompVolumeRendererImpl(int w, int h);

    ~CPUOffScreenCompVolumeRendererImpl();

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetRenderPolicy(CompRenderPolicy) override;

    auto GetBackendName() -> std::string override;

    void SetMPIRender(MPIRenderParameter) override;

    void SetStep(double step, int steps) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunc(TransferFunc tf) override;

    void render(bool sync) override;

    auto GetImage() -> const Image<Color4b> & override;

    void resize(int w, int h) override;

    void clear() override;

  private:
    int window_w, window_h;
    Image<Color4b> image;
    double step;
    double lod_dist[10];
    std::shared_ptr<CompVolume> comp_volume;
    uint32_t volume_dim_x, volume_dim_y, volume_dim_z;
    uint32_t volume_block_dim_x, volume_block_dim_y, volume_block_dim_z;
    double volume_space_x, volume_space_y, volume_space_z, base_space;
    uint32_t block_length, padding, no_padding_block_length, min_lod, max_lod;
    Camera camera;
    std::unique_ptr<BlockCacheManager<BlockArray9b>> block_cache_manager;
    Texture1D<Vec4f> tf_1d;
    Texture2D<Vec4f> tf_2d;
    std::unique_ptr<CDFManager> cdf_manager;
    int cdf_block_length;
    std::unordered_map<Vec4i, std::vector<uint32_t>> cdf_map;
    int cdf_dim_x, cdf_dim_y, cdf_dim_z;
    std::unordered_map<int, std::vector<uint32_t>> volume_value_map;
};

VS_END
