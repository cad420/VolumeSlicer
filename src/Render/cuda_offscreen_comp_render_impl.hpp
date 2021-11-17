//
// Created by wyz on 2021/9/13.
//

#pragma once

#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/volume_cache.hpp>

VS_START

class CUDAOffScreenCompVolumeRendererImpl: public CUDAOffScreenCompVolumeRenderer{
  public:
    CUDAOffScreenCompVolumeRendererImpl(int w,int h,CUcontext ctx);

    auto GetImage()->const Image<Color4b>& override;

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetRenderPolicy(CompRenderPolicy) override;

    auto GetBackendName()-> std::string override;

    void SetMPIRender(MPIRenderParameter) override;

    void SetStep(double step,int steps) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunc(TransferFunc tf) override;

    void render(bool sync) override;

    void resize(int w,int h) override;

    void clear() override;

  private:
    int window_w,window_h;
    Image<Color4b> image;
    CUcontext cu_context;
    float step;
    int steps;
    Camera camera;
    std::unique_ptr<CUDAVolumeBlockCache> volume_block_cache;
    std::shared_ptr<CompVolume> comp_volume;
};


VS_END