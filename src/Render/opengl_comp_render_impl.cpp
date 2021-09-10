//
// Created by wyz on 2021/7/30.
//
#include "opengl_comp_render_impl.hpp"
#include <random>


VS_START
std::unique_ptr<OpenGLCompVolumeRenderer> OpenGLCompVolumeRenderer::Create(int w, int h) {
    return std::make_unique<OpenGLCompVolumeRendererImpl>(w,h);
}
OpenGLCompVolumeRendererImpl::OpenGLCompVolumeRendererImpl(int w, int h)
:window_w(w),window_h(h)
{
    //init opengl context
    auto ins=GetModuleHandle(NULL);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::string idx=std::to_string(dist(rng));
    HWND window=create_window(ins,("wgl_invisable"+idx).c_str(),window_w,window_h);
    this->window_handle=GetDC(window);
    this->gl_context=create_opengl_context(this->window_handle);
    glEnable(GL_DEPTH_TEST);
    spdlog::info("successfully init OpenGL context.");
    //create shader

}

void OpenGLCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {

}

void OpenGLCompVolumeRendererImpl::SetCamera(Camera camera) {

}

void OpenGLCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf) {

}

void OpenGLCompVolumeRendererImpl::render() {

}

auto OpenGLCompVolumeRendererImpl::GetFrame() -> const Image<uint32_t> & {
    return image;
}

void OpenGLCompVolumeRendererImpl::resize(int w, int h) {

}

void OpenGLCompVolumeRendererImpl::clear() {

}
void OpenGLCompVolumeRendererImpl::SetMPIRender(MPIRenderParameter)
{

}
void OpenGLCompVolumeRendererImpl::SetStep(double step, int steps)
{

}
void OpenGLCompVolumeRendererImpl::calcMissedBlocks() {

}

void OpenGLCompVolumeRendererImpl::filterMissedBlocks() {

}

void OpenGLCompVolumeRendererImpl::sendRequests() {

}

void OpenGLCompVolumeRendererImpl::fetchBlocks() {

}


VS_END

