//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP

#include "cuda_comp_render_impl.cuh"
#include "Algorithm/helper_math.h"
#include "Common/cuda_utils.hpp"
using namespace CUDARenderer;

namespace {
    __constant__ CUDACompRenderParameter cudaCompRenderParameter;
    __constant__ CompVolumeParameter compVolumeParameter;
    __constant__ LightParameter lightParameter;
    __constant__ uint4 *mappingTable;
    __constant__ uint lodMappingTableOffset[10];
    __constant__ cudaTextureObject_t cacheVolumes[10];
    __constant__ cudaTextureObject_t transferFunc;
    __constant__ cudaTextureObject_t preIntTransferFunc;
    __constant__ uint* image;
    __constant__ uint* missedBlocks;
    uint *d_image = nullptr;
    int image_w = 0, image_h = 0;
    uint4 *d_mappingTable = nullptr;
    cudaArray* tf=nullptr;
    cudaArray* preInt_tf=nullptr;
    uint* d_missedBlocks=nullptr;


    __device__ void VirtualSample(){

    }
    __device__ uint rgbaFloatToUInt(float4 rgba)
    {
        rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
        rgba.y = __saturatef(rgba.y);
        rgba.z = __saturatef(rgba.z);
        rgba.w = __saturatef(rgba.w);
        return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
    }
    __device__ int evaluateLod(float distance){
//        return log2f(distance/0.2f+1);
        if(distance<0.2f){
            return 0;
        }
        else if(distance<0.6f){
            return 1;
        }
        else if(distance<1.2f){
            return 2;
        }
        else if(distance<2.4f){
            return 3;
        }
        else if(distance<4.8f){
            return 4;
        }
        else if(distance<9.6f){
            return 5;
        }
        else{
            return 6;
        }
    }

    /*
     * using raycast to calculate intersect lod blocks
     */
    __global__ void CUDACalcBlockKernel(){
        uint32_t image_x=blockIdx.x*blockDim.x+threadIdx.x;
        uint32_t image_y=blockIdx.y*blockDim.y+threadIdx.y;
        if(image_x>=cudaCompRenderParameter.w || image_y>=cudaCompRenderParameter.h) return;
        float x_offset=(image_x-cudaCompRenderParameter.w/2)*2.f/cudaCompRenderParameter.w
                *tanf(cudaCompRenderParameter.fov/2)*cudaCompRenderParameter.w/cudaCompRenderParameter.h;
        float y_offset=(image_y-cudaCompRenderParameter.h/2)*2.f/cudaCompRenderParameter.h
                *tanf(cudaCompRenderParameter.fov/2);
        float3 pixel_view_pos=cudaCompRenderParameter.view_pos
                +cudaCompRenderParameter.view_direction
                +x_offset*cudaCompRenderParameter.right
                -y_offset*cudaCompRenderParameter.up;
        float3 ray_direction=normalize(pixel_view_pos-cudaCompRenderParameter.view_pos);
        float3 start_pos=cudaCompRenderParameter.view_pos;
        float3 ray_pos=start_pos;
        int last_lod=0;
        float cur_step=cudaCompRenderParameter.step*1;
        int3 block_dim=compVolumeParameter.block_dim;
        int no_padding_block_length=compVolumeParameter.no_padding_block_length;
        while(true){
            int cur_lod=evaluateLod(length(ray_pos-start_pos));
            if(cur_lod>last_lod){
                cur_step*=2;
                block_dim=(block_dim+1) / 2;
                no_padding_block_length*2;
            }
            if(cur_lod>6)
                break;
            int3 block_idx=make_int3(ray_pos/cudaCompRenderParameter.space/no_padding_block_length);
            size_t flat_block_idx=block_idx.x*block_dim.x*block_dim.y
                    +block_idx.y*block_dim.x
                    +block_idx.z;//+lodMappingTableOffset[cur_lod];
            if(missedBlocks[flat_block_idx]==0){
                atomicExch(&missedBlocks[flat_block_idx],1);
            }

            ray_pos+=ray_direction*cur_step;
            if(ray_pos.x<0.f || ray_pos.x>compVolumeParameter.volume_board.x
            || ray_pos.y<0.f || ray_pos.y>compVolumeParameter.volume_board.y
            || ray_pos.z<0.f || ray_pos.z>compVolumeParameter.volume_board.z){
                break;
            }
        }

    }
    __global__ void CUDARenderKernel(){
        uint32_t image_x=blockIdx.x*blockDim.x+threadIdx.x;
        uint32_t image_y=blockIdx.y*blockDim.y+threadIdx.y;
        if(image_x>=cudaCompRenderParameter.w || image_y>=cudaCompRenderParameter.h) return;
        uint64_t image_idx=(uint64_t)image_y*cudaCompRenderParameter.w+image_x;


        image[image_idx]=rgbaFloatToUInt(make_float4(image_x*1.f/cudaCompRenderParameter.w,
                                         image_y*1.f/cudaCompRenderParameter.h,
                                         0.f,1.f));
    }

}//end of namespace

static void CreateDeviceTransferFunc(){

}
static void CreateDevicePreIntTransferFunc(){

}
static void CreateDeviceRenderImage(uint32_t w,uint32_t h){
    if(d_image){
        CUDA_RUNTIME_API_CALL(cudaFree(d_image));
    }
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_image,(size_t)w*h*sizeof(uint)));
    image_w=w;
    image_h=h;
    assert(d_image);
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(image,&d_image,sizeof(uint*)));
}
static void CreateDeviceMappingTable(){

}
static void CreateDeviceMissedBlocks(size_t size){
    assert(!d_missedBlocks);
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_missedBlocks,size*sizeof(uint32_t)));
    CUDA_RUNTIME_API_CALL(cudaMemset(d_missedBlocks,0,size*sizeof(uint32_t)));
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(missedBlocks,&d_missedBlocks,sizeof(uint*)));
}

namespace CUDARenderer{


void UploadTransferFunc(float *data, size_t size) {
    if(!tf){
        CreateCUDATexture1D(256,&tf,&transferFunc);
    }
    UpdateCUDATexture1D((uint8_t*)data,tf,256*4,0);
}



void UploadPreIntTransferFunc(float *data, size_t size) {
    if(!preInt_tf){
        CreateCUDATexture2D(256,256,&preInt_tf,&preIntTransferFunc);
    }
    UpdateCUDATexture2D((uint8_t*)data,preInt_tf,256*4,256,0,0);
}

void CUDACalcBlock(uint32_t *missed_blocks, size_t size,uint32_t w,uint32_t h) {
    if(!d_missedBlocks){
        CreateDeviceMissedBlocks(size);
    }
    CUDA_RUNTIME_API_CALL(cudaMemset(d_missedBlocks,0,size*sizeof(uint32_t)));

    dim3 threads_per_block={16,16};
    dim3 blocks_per_grid={(w+threads_per_block.x-1)/threads_per_block.x,(h+threads_per_block.y-1)/threads_per_block.y};

    CUDACalcBlockKernel<<<blocks_per_grid,threads_per_block>>>();
    CUDA_RUNTIME_CHECK

    CUDA_RUNTIME_API_CALL(cudaMemcpy(missed_blocks,d_missedBlocks,size*sizeof(uint32_t),cudaMemcpyDeviceToHost));
}

void CUDARender(uint32_t w,uint32_t h,uint32_t *image) {
    if(image_w!=w || image_h!=h){
        CreateDeviceRenderImage(w,h);
    }

    dim3 threads_per_block={16,16};
    dim3 blocks_per_grid={(w+threads_per_block.x-1)/threads_per_block.x,(h+threads_per_block.y-1)/threads_per_block.y};

    CUDARenderKernel<<<blocks_per_grid,threads_per_block>>>();
    CUDA_RUNTIME_CHECK

    CUDA_RUNTIME_API_CALL(cudaMemcpy(image,d_image,(size_t)w*h*sizeof(uint),cudaMemcpyDeviceToHost));
}

void UploadLightParameter(const LightParameter &light) {
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(lightParameter,&light,sizeof(LightParameter)));
}

void UploadCompVolumeParameter(const CompVolumeParameter &comp_volume) {
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(compVolumeParameter,&comp_volume,sizeof(CompVolumeParameter)));
}

void UploadCUDACompRenderParameter(const CUDACompRenderParameter &comp_render) {
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cudaCompRenderParameter,&comp_render,sizeof(CUDACompRenderParameter)));
}

void SetCUDATextureObject(cudaTextureObject_t *textures, size_t size) {
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cacheVolumes,textures,size*sizeof(cudaTextureObject_t)));
}



}

#endif //VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
