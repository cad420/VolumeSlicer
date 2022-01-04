#include "comp_volume_sample.cuh"
#include "Common/helper_math.h"
#include "Common/cuda_utils.hpp"
VS_START
__constant__ CompSampleParameter compSampleParameter;
__constant__ BlockParameter blockParameter;
//__constant__ variable's size should be known when compile
uint4* h_mappingTable=nullptr;
uint4* d_mappingTable=nullptr;
size_t mappingTableSize=0;
__constant__ uint4* mappingTable;
__constant__ uint lodMappingTableOffset[10];//!max lod is 9 and offset is for uint4 not uint
__constant__ cudaTextureObject_t cacheVolumes[10];//max texture num is 10


//sample_pos must in voxel world-coord
__device__ float VirtualSample(float3 sample_pos){
    int lod_t=PowII(2,compSampleParameter.lod);
    int lod_no_padding_block_length=blockParameter.no_padding_block_length*lod_t;
    int3 virtual_block_index=make_int3(sample_pos / lod_no_padding_block_length);
    int3 lod_block_dim=make_int3(make_float3(blockParameter.block_dim + lod_t-1) / lod_t);
//    if(sample_pos.x<0.f
//       || sample_pos.y<0.f
//       || sample_pos.z<0.f ){
//        return 1.0f;
//    }
    if(sample_pos.x<0.f || virtual_block_index.x>=lod_block_dim.x
    || sample_pos.y<0.f || virtual_block_index.y>=lod_block_dim.y
    || sample_pos.z<0.f || virtual_block_index.z>=lod_block_dim.z){
        return 0.f;
    }
    //get offset in physical length's block
    float3 offset_in_no_padding_block=(sample_pos-make_float3(virtual_block_index*lod_no_padding_block_length))/lod_t;

    size_t flat_virtual_block_index=virtual_block_index.z*lod_block_dim.y*lod_block_dim.x
            +virtual_block_index.y*lod_block_dim.x+virtual_block_index.x
            +lodMappingTableOffset[compSampleParameter.lod];

    uint4 physical_block_index=mappingTable[flat_virtual_block_index];

    uint physical_block_flag=(physical_block_index.w>>16)&(0x0000ffff);

    if(physical_block_flag!=1){
        return 0.f;
    }

    float3 physical_sample_pos=make_float3(physical_block_index.x,physical_block_index.y,physical_block_index.z)
            *blockParameter.block_length+offset_in_no_padding_block+make_float3(blockParameter.padding);

    physical_sample_pos /= make_float3(blockParameter.texture_size3);
    uint tex_id=physical_block_index.w&(0x0000ffff);
    return tex3D<float>(cacheVolumes[tex_id],physical_sample_pos.x,physical_sample_pos.y,physical_sample_pos.z);
}

__global__ void CUDACompVolumeSample(uint8_t* image){
    int image_x=blockIdx.x*blockDim.x+threadIdx.x;
    int image_y=blockIdx.y*blockDim.y+threadIdx.y;
    if(image_x>=compSampleParameter.image_w || image_y>=compSampleParameter.image_h) return;
    uint64_t image_idx=(uint64_t)image_y*compSampleParameter.image_w+image_x;


    float3 virtual_sample_pos;
    virtual_sample_pos=compSampleParameter.origin+((image_x-compSampleParameter.image_w/2)*compSampleParameter.voxels_per_pixel.x*compSampleParameter.right
            +(image_y-compSampleParameter.image_h/2)*compSampleParameter.voxels_per_pixel.y*compSampleParameter.down)
                    /compSampleParameter.space_ratio;
//    virtual_sample_pos=virtual_sample_pos*0.01f/compSampleParameter.space;
    int stop_step=0,start_step=0;
    if(compSampleParameter.direction & 0b1){
        stop_step = compSampleParameter.steps;
    }
    if(compSampleParameter.direction & 0b10){
        start_step = - compSampleParameter.steps;
    }
    float max_scalar = 0.f;
    for(int i_step = start_step;i_step <= stop_step; i_step++){
        float3 sample_pos = virtual_sample_pos + (i_step * compSampleParameter.step * compSampleParameter.normal/compSampleParameter.space);
        float sample_scalar = VirtualSample(sample_pos);
        if(sample_scalar>max_scalar) max_scalar = sample_scalar;
    }

    image[image_idx] = max_scalar * 255;
//    image[image_idx]=VirtualSample(virtual_sample_pos)*255;
//    image[image_idx]=128;
}

void SetCUDASampleParameter(const CompSampleParameter& parameter){
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(compSampleParameter,&parameter,sizeof(CompSampleParameter)));
}
void SetCUDABlockParameter(const BlockParameter& parameter){
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(blockParameter,&parameter,sizeof(BlockParameter)));
}

void SetCUDATextureObject(cudaTextureObject_t* textures,size_t size){
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cacheVolumes,textures,size*sizeof(cudaTextureObject_t)));
}

static void CreateDeviceMappingTable(const uint32_t*data ,size_t size){
    assert(!d_mappingTable && !mappingTableSize);
    mappingTableSize=size/4;
    CUDA_RUNTIME_API_CALL(cudaMallocHost(&h_mappingTable,size*sizeof(size_t)));
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_mappingTable,size*sizeof(size_t)));
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(mappingTable,&d_mappingTable,sizeof(uint4*)));
}

std::vector<std::array<uint32_t, 4>> GetCUDAUnUploadBlocks(){
    CUDA_RUNTIME_API_CALL(cudaMemcpy(h_mappingTable,d_mappingTable,mappingTableSize*sizeof(uint4),cudaMemcpyDeviceToHost));
    std::vector<std::array<uint32_t,4>> blocks;
    for(size_t i=0;i<mappingTableSize;i++){
        if(((h_mappingTable[i].w>>20)&0x1)==1){
                        spdlog::critical("{0} {1} {2} {3} {4:x}.",i,h_mappingTable[i].x,
                         h_mappingTable[i].y,h_mappingTable[i].z,h_mappingTable[i].w);
            blocks.push_back({h_mappingTable[i].x,
                                 h_mappingTable[i].y,
                                 h_mappingTable[i].z,
                                 h_mappingTable[i].w&0x0000ffff});
        }
//        else if(h_mappingTable[i].w!=0){
//            spdlog::info("{0} {1} {2} {3} {4}.",i,h_mappingTable[i].x,
//                         h_mappingTable[i].y,h_mappingTable[i].z,h_mappingTable[i].w);
//        }
    }
    return blocks;
}


/*************************************************************************************************************
 *
 */

void CUDACompVolumeSampler::UploadCompSampleParameter(const CompSampleParameter &sampleParameter) {
    SetCUDACtx();
    SetCUDASampleParameter(sampleParameter);
}

void CUDACompVolumeSampler::UploadBlockParameter(const BlockParameter & blockParameter) {
    SetCUDACtx();
    SetCUDABlockParameter(blockParameter);
}


void CUDACompVolumeSampler::Sample(uint8_t *data, Slice slice) {
    SetCUDACtx();
    int w=slice.n_pixels_width;
    int h=slice.n_pixels_height;
    if(w!=old_w || h!=old_h){
        if(cu_sample_result)
            CUDA_RUNTIME_API_CALL(cudaFree(cu_sample_result));
        CUDA_RUNTIME_API_CALL(cudaMalloc((void**)&cu_sample_result,(size_t)w*h));
    }
    assert(cu_sample_result);
    old_w=w;
    old_h=h;

    dim3 threads_per_block={16,16};
    dim3 blocks_per_grid={(w+threads_per_block.x-1)/threads_per_block.x,(h+threads_per_block.y-1)/threads_per_block.y};

    CUDACompVolumeSample<<<blocks_per_grid,threads_per_block>>>(cu_sample_result);
    CUDA_RUNTIME_CHECK

    CUDA_RUNTIME_API_CALL(cudaMemcpy(data,cu_sample_result,(size_t)w*h,cudaMemcpyDeviceToHost));
//    spdlog::info("Finish CUDA comp volume sample.");
}

void CUDACompVolumeSampler::SetBlockInfo(uint32_t block_length,uint32_t padding) {
    this->block_length=block_length;
    this->padding=padding;
}

std::vector<std::array<uint32_t, 4>> CUDACompVolumeSampler::GetUnUploadBlocks() {
    return GetCUDAUnUploadBlocks();
}

void CUDACompVolumeSampler::UploadMappingTable(const uint32_t *data, size_t size) {
    SetCUDACtx();
    if(!d_mappingTable){
        CreateDeviceMappingTable(data,size);
    }
    CUDA_RUNTIME_API_CALL(cudaMemcpy(d_mappingTable,data,size*sizeof(uint32_t),cudaMemcpyHostToDevice));

}

void CUDACompVolumeSampler::UploadLodMappingTableOffset(const uint32_t *data, size_t size) {
    SetCUDACtx();
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(lodMappingTableOffset,data,size*sizeof(uint32_t)));
}

void CUDACompVolumeSampler::SetCUDATextureObject(cudaTextureObject_t *textures, size_t size) {
    SetCUDACtx();
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cacheVolumes,textures,size*sizeof(cudaTextureObject_t)));
}

CUDACompVolumeSampler::~CUDACompVolumeSampler() {
    spdlog::info("Call ~CUDACompVolumeSampler destructor.");
    cudaFree(cu_sample_result);
}


VS_END

