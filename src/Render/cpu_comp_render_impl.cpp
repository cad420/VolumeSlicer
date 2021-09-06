//
// Created by wyz on 2021/7/30.
//
#include "cpu_comp_render_impl.hpp"
VS_START
std::unique_ptr<CPUCompVolumeRenderer> CPUCompVolumeRenderer::Create(int w, int h) {
    return std::unique_ptr<CPUCompVolumeRenderer>();
}

CPUCompVolumeRendererImpl::CPUCompVolumeRendererImpl(int w, int h) {

}

void CPUCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {

}


void CPUCompVolumeRendererImpl::SetCamera(Camera camera) {

}

void CPUCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf) {

}

void CPUCompVolumeRendererImpl::render() {

}

auto CPUCompVolumeRendererImpl::GetFrame() -> const Image<uint32_t> & {
    return image;
}

void CPUCompVolumeRendererImpl::resize(int w, int h) {

}

void CPUCompVolumeRendererImpl::clear() {

}
void CPUCompVolumeRendererImpl::SetMPIRender(MPIRenderParameter)
{

}
void CPUCompVolumeRendererImpl::SetStep(double step, int steps)
{

}

VS_END

