//
// Created by wyz on 2021/8/23.
//
#include <VolumeSlicer/cuda_context.hpp>
#include <VolumeSlicer/singleton.hpp>

void SetCUDACtx(CUdevice d){
    Singleton<CUDACtx>::init(d);
}
CUcontext GetCUDACtx(){
    return Singleton<CUDACtx>::get()->GetCUDACtx();
}
size_t GetCUDAFreeMem(){
    size_t free,total;
    cudaMemGetInfo(&free,&total);
    return free;
}
size_t GetCUDAUsedMem(){
    size_t free,total;
    cudaMemGetInfo(&free,&total);
    return total-free;
}