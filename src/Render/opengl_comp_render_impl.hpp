//
// Created by wyz on 2021/7/30.
//

#ifndef VOLUMESLICER_OPENGL_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_OPENGL_COMP_RENDER_IMPL_HPP

#include <VolumeSlicer/render.hpp>
#include "Render/wgl_wrap.hpp"
VS_START


class OpenGLCompVolumeRendererImpl:public OpenGLCompVolumeRenderer{
public:
    OpenGLCompVolumeRendererImpl(int w,int h);

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetRenderPolicy(CompRenderPolicy) override;

    void SetMPIRender(MPIRenderParameter) override ;

    void SetStep(double step,int steps) override;

    void SetCamera(Camera camera) override ;

    void SetTransferFunc(TransferFunc tf) override ;

    void render() override ;

    auto GetFrame()  -> const Image<uint32_t>&  override ;

    void resize(int w,int h) override ;

    void clear() override ;

private:
    void calcMissedBlocks();

    void filterMissedBlocks();

    void sendRequests();

    void fetchBlocks();

private:
    void setCurrentCtx(){wglMakeCurrent(window_handle,gl_context);}
private:
    Image<uint32_t> image;
    HDC window_handle;
    HGLRC gl_context;
    int window_w,window_h;

};


VS_END

#endif //VOLUMESLICER_OPENGL_COMP_RENDER_IMPL_HPP
