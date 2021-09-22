//
// Created by wyz on 2021/7/30.
//

#ifndef VOLUMESLICER_CPU_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_CPU_COMP_RENDER_IMPL_HPP

#include <VolumeSlicer/render.hpp>
#include <Utils/block_cache.hpp>
#include "Render/Texture/sampler.hpp"
VS_START

class CPUOffScreenCompVolumeRendererImpl: public CPUOffScreenCompVolumeRenderer{
public:
    CPUOffScreenCompVolumeRendererImpl(int w,int h);
    ~CPUOffScreenCompVolumeRendererImpl();
    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetRenderPolicy(CompRenderPolicy) override;

    void SetMPIRender(MPIRenderParameter) override ;

    void SetStep(double step,int steps) override;

    void SetCamera(Camera camera) override ;

    void SetTransferFunc(TransferFunc tf) override ;

    void render() override ;

    auto GetFrame()  -> const Image<uint32_t>&  override ;

    auto GetImage()->const Image<Color4b>& override;

    void resize(int w,int h) override ;

    void clear() override ;
private:
    int window_w,window_h;
    Image<uint32_t> frame;
    Image<Color4b > image;
    double step;
    std::shared_ptr<CompVolume> comp_volume;
    uint32_t volume_dim_x,volume_dim_y,volume_dim_z;
    double volume_space_x,volume_space_y,volume_space_z,base_space;
    uint32_t block_length,padding,no_padding_block_length,min_lod,max_lod;
    Camera camera;
    std::unique_ptr<BlockCacheManager<BlockArray9b>> block_cache_manager;
    Texture1D<Vec4f> tf_1d;
    Texture2D<Vec4f> tf_2d;
};

VS_END

#endif //VOLUMESLICER_CPU_COMP_RENDER_IMPL_HPP
