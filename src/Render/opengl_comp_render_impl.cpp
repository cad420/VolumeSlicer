//
// Created by wyz on 2021/7/30.
//
#include "opengl_comp_render_impl.hpp"
#include <random>
#include <Common/gl_helper.hpp>
#include <iostream>
VS_START
std::unique_ptr<OpenGLCompVolumeRenderer> OpenGLCompVolumeRenderer::Create(int w, int h) {
    return std::make_unique<OpenGLCompVolumeRendererImpl>(w,h);
}
OpenGLCompVolumeRendererImpl::OpenGLCompVolumeRendererImpl(int w, int h)
:window_w(w),window_h(h)
{
    //init opengl context
    if (glfwInit() == GLFW_FALSE)
    {
        std::cout << "Failed to init GLFW" << std::endl;
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, true);
    window=glfwCreateWindow(window_w,window_h,"HideWindow",nullptr, nullptr);
    if(window==nullptr){
        throw std::runtime_error("Create GLFW window failed.");
    }
    setCurrentCtx();
    glfwHideWindow(window);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        throw std::runtime_error("GLAD failed to load opengl api");
    }
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
    GL_CHECK
    spdlog::info("successfully init OpenGL context.");
    //create shader
    OpenGLCompVolumeRendererImpl::resize(w,h);

}

void OpenGLCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {

}

void OpenGLCompVolumeRendererImpl::SetCamera(Camera camera) {

}

void OpenGLCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf) {

}

void OpenGLCompVolumeRendererImpl::render() {
    setCurrentCtx();
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glClearColor(0.0f,0.f,0.f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glFinish();
    GL_CHECK

}

auto OpenGLCompVolumeRendererImpl::GetFrame() -> const Image<uint32_t> & {
    setCurrentCtx();
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glReadPixels(0,0,window_w,window_h,GL_RGBA,GL_UNSIGNED_BYTE,reinterpret_cast<void*>(image.data.data()));

    return image;
}

void OpenGLCompVolumeRendererImpl::resize(int w, int h) {
    setCurrentCtx();
    this->image.width=w;
    this->image.height=h;
    this->image.data.resize(w*h,0);
    glViewport(0,0,w,h);
}

void OpenGLCompVolumeRendererImpl::clear() {

}
void OpenGLCompVolumeRendererImpl::SetRenderPolicy(CompRenderPolicy)
{

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

