//
// Created by wyz on 2021/8/23.
//
#include <VolumeSlicer/cuda_context.hpp>
#include <VolumeSlicer/singleton.hpp>

// global variable between dll and main program
//https://myprogrammingnotes.com/global-variable-main-program-conflict-global-variable-dll-name.html

//using cuda driver api in different dlls should explict create cuda context in the dll
//todo
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