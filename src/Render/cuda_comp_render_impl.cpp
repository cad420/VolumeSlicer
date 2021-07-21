//
// Created by wyz on 2021/7/21.
//
#include "cuda_comp_render_impl.hpp"
#include "cuda_comp_render_impl.cuh"
#include
VS_START
std::unique_ptr<CUDACompVolumeRenderer> CUDACompVolumeRenderer::Create(int w, int h, CUcontext ctx) {
    return std::make_unique<CUDACompVolumeRendererImpl>(w,h,ctx);
}

CUDACompVolumeRendererImpl::CUDACompVolumeRendererImpl(int w, int h, CUcontext ctx)
:window_w(w),window_h(h)
{
    if(ctx){
        this->cu_context=ctx;
    }
    else{
        this->cu_context=GetCUDACtx();
        if(!cu_context)
            throw std::runtime_error("cu_context is nullptr");
    }
    this->image.data.resize((size_t)w*h,0);
}

void CUDACompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {
    this->comp_volume=comp_volume;
}

void CUDACompVolumeRendererImpl::SetCamera(Camera camera) {
    this->camera=camera;
}

void CUDACompVolumeRendererImpl::SetTransferFunction(TransferFunc tf) {

}

void CUDACompVolumeRendererImpl::render() {
    assert(image.data.size()==(size_t)window_h*window_w);

    CUDARenderer::CUDARender(image.data.data());

}

auto CUDACompVolumeRendererImpl::GetFrame() -> Image<uint32_t> {
    return {};
}

void CUDACompVolumeRendererImpl::resize(int w, int h) {

}

void CUDACompVolumeRendererImpl::clear() {

}


VS_END



