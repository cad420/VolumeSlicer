//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/volume_cache.hpp>

VS_START
class CUDACompVolumeRendererImpl: public CUDACompVolumeRenderer{
public:
    CUDACompVolumeRendererImpl(int w,int h,CUcontext ctx);

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunction(TransferFunc tf) override;

    void render() override;

    auto GetFrame()->Image<uint32_t> override;

    void resize(int w,int h) override;

    void clear() override;
private:
    int window_w,window_h;
    CUcontext cu_context;
    std::shared_ptr<CompVolume> comp_volume;
    Camera camera;
    Image<uint32_t> image;
};


VS_END
#endif //VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
