//
// Created by wyz on 2021/7/30.
//
#include "opengl_comp_render_impl.hpp"


VS_START
std::unique_ptr<OpenGLCompVolumeRenderer> OpenGLCompVolumeRenderer::Create(int w, int h) {
    return std::make_unique<OpenGLCompVolumeRendererImpl>(w,h);
}
OpenGLCompVolumeRendererImpl::OpenGLCompVolumeRendererImpl(int w, int h) {

}

void OpenGLCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {

}

void OpenGLCompVolumeRendererImpl::SetMPIViewOffset(float, float)
{

}

void OpenGLCompVolumeRendererImpl::SetCamera(vs::Camera camera) {

}

void OpenGLCompVolumeRendererImpl::SetTransferFunc(vs::TransferFunc tf) {

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

VS_END

