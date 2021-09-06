//
// Created by wyz on 2021/7/30.
//

#ifndef VOLUMESLICER_OPENGL_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_OPENGL_COMP_RENDER_IMPL_HPP

#include <VolumeSlicer/render.hpp>

VS_START


class OpenGLCompVolumeRendererImpl:public OpenGLCompVolumeRenderer{
public:
    OpenGLCompVolumeRendererImpl(int w,int h);

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetMPIViewOffset(float,float) override;

    void SetCamera(Camera camera) override ;

    void SetTransferFunc(TransferFunc tf) override ;

    void render() override ;

    auto GetFrame()  -> const Image<uint32_t>&  override ;

    void resize(int w,int h) override ;

    void clear() override ;
private:
    Image<uint32_t> image;
};


VS_END

#endif //VOLUMESLICER_OPENGL_COMP_RENDER_IMPL_HPP
