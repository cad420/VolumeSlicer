//
// Created by wyz on 2021/7/21.
//
#include "cuda_comp_render_impl.hpp"
#include "cuda_comp_render_impl.cuh"
#include "Render/transfer_function_impl.hpp"
#include "Algorithm/helper_math.h"
#include <chrono>
#include <iostream>
#define START_CPU_TIMER \
{auto _start=std::chrono::steady_clock::now();

#define END_CPU_TIMER \
auto _end=std::chrono::steady_clock::now();\
auto _t=std::chrono::duration_cast<std::chrono::milliseconds>(_end-_start);\
std::cout<<"CPU cost time : "<<_t.count()<<"ms"<<std::endl;}


#define START_CUDA_DRIVER_TIMER \
CUevent start,stop;\
float elapsed_time;\
cuEventCreate(&start,CU_EVENT_DEFAULT);\
cuEventCreate(&stop,CU_EVENT_DEFAULT);\
cuEventRecord(start,0);

#define STOP_CUDA_DRIVER_TIMER \
cuEventRecord(stop,0);\
cuEventSynchronize(stop);\
cuEventElapsedTime(&elapsed_time,start,stop);\
cuEventDestroy(start);\
cuEventDestroy(stop);\
std::cout<<"GPU cost time: "<<elapsed_time<<"ms"<<std::endl;


#define START_CUDA_RUNTIME_TIMER \
{cudaEvent_t     start, stop;\
float   elapsedTime;\
(cudaEventCreate(&start)); \
(cudaEventCreate(&stop));\
(cudaEventRecord(start, 0));

#define STOP_CUDA_RUNTIME_TIMER \
(cudaEventRecord(stop, 0));\
(cudaEventSynchronize(stop));\
(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("\tGPU cost time used: %.f ms\n", elapsedTime);\
(cudaEventDestroy(start));\
(cudaEventDestroy(stop));}

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
    CUDACompVolumeRendererImpl::resize(w,h);
}

void CUDACompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume) {
    this->comp_volume=comp_volume;

    this->cuda_volume_block_cache=CUDAVolumeBlockCache::Create();
    this->cuda_volume_block_cache->SetCacheBlockLength(comp_volume->GetBlockLength()[0]);
    this->cuda_volume_block_cache->SetCacheCapacity(6,2048,1024,1024);
    this->cuda_volume_block_cache->CreateMappingTable(this->comp_volume->GetBlockDim());
    this->missed_blocks_pool.resize(this->cuda_volume_block_cache->GetMappingTable().size(),0);

    CUDARenderer::CompVolumeParameter compVolumeParameter;
    auto block_length=comp_volume->GetBlockLength();
    compVolumeParameter.block_length=block_length[0];
    compVolumeParameter.padding=block_length[1];
    compVolumeParameter.no_padding_block_length=block_length[0]-2*block_length[1];
    auto block_dim=comp_volume->GetBlockDim(0);
    compVolumeParameter.block_dim=make_int3(block_dim[0],block_dim[1],block_dim[2]);
    compVolumeParameter.texture_shape=make_int4(3,2048,1024,1024);
    compVolumeParameter.volume_board=make_int3(comp_volume->GetVolumeDimX()*comp_volume->GetVolumeSpaceX() * compVolumeParameter.no_padding_block_length,
                                               comp_volume->GetVolumeDimY()*comp_volume->GetVolumeSpaceY() * compVolumeParameter.no_padding_block_length,
                                               comp_volume->GetVolumeDimZ()*comp_volume->GetVolumeSpaceZ() * compVolumeParameter.no_padding_block_length
                                               );
    CUDARenderer::UploadCompVolumeParameter(compVolumeParameter);
}

void CUDACompVolumeRendererImpl::SetCamera(Camera camera) {
    this->camera=camera;
}

void CUDACompVolumeRendererImpl::SetTransferFunction(TransferFunc tf) {
    TransferFuncImpl tf_impl(tf);
    CUDARenderer::UploadTransferFunc(tf_impl.getTransferFunction().data());
    CUDARenderer::UploadPreIntTransferFunc(tf_impl.getPreIntTransferFunc().data());

    //todo move to another place
    CUDARenderer::LightParameter lightParameter;
    lightParameter.bg_color=make_float4(0.f,0.f,0.f,0.f);
    lightParameter.ka=0.5f;
    lightParameter.kd=0.7f;
    lightParameter.ks=0.5f;
    lightParameter.shininess=64.f;
    CUDARenderer::UploadLightParameter(lightParameter);
}

void CUDACompVolumeRendererImpl::render() {
    assert(image.data.size()==(size_t)window_h*window_w);

    //may change every time render
    CUDARenderer::CUDACompRenderParameter cudaCompRenderParameter;
    cudaCompRenderParameter.w=window_w;
    cudaCompRenderParameter.h=window_h;
    cudaCompRenderParameter.fov=45.f;
    cudaCompRenderParameter.step=0.00016f;
    cudaCompRenderParameter.view_pos=make_float3(camera.pos[0],camera.pos[1],camera.pos[2]);
    cudaCompRenderParameter.view_direction=normalize(make_float3(camera.look_at[0]-camera.pos[0],
                                                       camera.look_at[1]-camera.pos[1],
                                                       camera.look_at[2]-camera.pos[2]));
    cudaCompRenderParameter.up=make_float3(camera.up[0],camera.up[1],camera.up[2]);
    cudaCompRenderParameter.right=make_float3(camera.right[0],camera.right[1],camera.right[2]);
    cudaCompRenderParameter.space=make_float3(comp_volume->GetVolumeSpaceX(),
                                              comp_volume->GetVolumeSpaceY(),
                                              comp_volume->GetVolumeSpaceZ());
    CUDARenderer::UploadCUDACompRenderParameter(cudaCompRenderParameter);

    START_CUDA_RUNTIME_TIMER
    CUDARenderer::CUDACalcBlock(missed_blocks_pool.data(),missed_blocks_pool.size(),window_w,window_h);
    STOP_CUDA_RUNTIME_TIMER

    int cnt=0;
    for(auto i=0;i<missed_blocks_pool.size();i++){
        if(missed_blocks_pool[i]!=0){
//            std::cout<<i<<" "<<missed_blocks_pool[i]<<std::endl;
            cnt++;
        }
    }
    std::cout<<"cnt: "<<cnt<<std::endl;

    CUDARenderer::CUDARender(window_w,window_h,image.data.data());

}

auto CUDACompVolumeRendererImpl::GetFrame() -> const Image<uint32_t>& {
    return image;
}

void CUDACompVolumeRendererImpl::resize(int w, int h) {
    if(w<0 || h<0 || w>10000 || h>10000){
        spdlog::error("error w({0}) or h({1}) for cuda comp volume renderer.",w,h);
        return;
    }
    this->window_w=w;
    this->window_h=h;
    this->image.data.resize((size_t)w*h,0);

}

void CUDACompVolumeRendererImpl::clear() {

}


VS_END



