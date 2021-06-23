#include"Algorithm/raw_volume_sample.cuh"
#include"Algorithm/helper_math.h"
VS_START



__constant__ RawSampleParameter sampleParameter;//use for kernel function


__global__ void CUDARawVolumeSample(uint8_t* image,//output result
                                    cudaTextureObject_t  volume_data//cuda texture for volume data

                                    );


void CUDARawVolumeSampler::SetVolumeData(uint8_t *data, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z) {
    assert(data && dim_x && dim_y && dim_z);
    this->volume_x=dim_x;
    this->volume_y=dim_y;
    this->volume_z=dim_z;
    this->volume_data_size=(size_t)dim_x*dim_y*dim_z;

    auto volume_dim=make_cudaExtent(dim_x,dim_y,dim_z);
    auto channelDesc=cudaCreateChannelDesc<uint8_t>();
    CUDA_RUNTIME_API_CALL(cudaMalloc3DArray(&cu_volume_data,&channelDesc,volume_dim));
    cudaResourceDesc texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));
    texRes.resType=cudaResourceTypeArray;
    texRes.res.array.array=cu_volume_data;
    cudaTextureDesc texDesc;
    memset(&texDesc,0,sizeof(cudaTextureDesc));
    texDesc.normalizedCoords=true;
    texDesc.filterMode=cudaFilterModeLinear;
    texDesc.borderColor[0]=0.f;
    texDesc.borderColor[1]=0.f;
    texDesc.borderColor[2]=0.f;
    texDesc.borderColor[3]=0.f;
    texDesc.addressMode[0]=cudaAddressModeBorder;
    texDesc.addressMode[1]=cudaAddressModeBorder;
    texDesc.addressMode[2]=cudaAddressModeBorder;
    texDesc.readMode=cudaReadModeNormalizedFloat;
    CUDA_RUNTIME_API_CALL(cudaCreateTextureObject(&volume_texture,&texRes,&texDesc, nullptr));

    CUDA_MEMCPY3D m={0};
    m.srcMemoryType=CU_MEMORYTYPE_HOST;
    m.srcHost=data;

    m.dstMemoryType=CU_MEMORYTYPE_ARRAY;
    m.dstArray=(CUarray)cu_volume_data;
    m.dstXInBytes=0;
    m.dstY=0;
    m.dstZ=0;

    m.WidthInBytes=dim_x;
    m.Height=dim_y;
    m.Depth=dim_z;
    CUDA_DRIVER_API_CALL(cuMemcpy3D(&m));

    spdlog::info("Successfully set volume data to CUDA.");
}


void CUDARawVolumeSampler::Sample(uint8_t *data, Slice slice,float space_x,float space_y,float space_z) {
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

    RawSampleParameter sample_parameter;
    sample_parameter.image_w=w;
    sample_parameter.image_h=h;
    sample_parameter.origin=make_float3(slice.origin[0],slice.origin[1],slice.origin[2]);
    sample_parameter.right=make_float3(slice.right[0],slice.right[1],slice.right[2]);
    sample_parameter.down=make_float3(-slice.up[0],-slice.up[1],-slice.up[2]);
    sample_parameter.voxels_per_pixel=make_float2(slice.voxel_per_pixel_width,slice.voxel_per_pixel_height);
    sample_parameter.volume_board=make_float3(volume_x*space_x,volume_y*space_y,volume_z*space_z);
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(sampleParameter,&sample_parameter,sizeof(RawSampleParameter)));

    dim3 threads_per_block={16,16};
    dim3 blocks_per_grid={(w+threads_per_block.x-1)/threads_per_block.x,(h+threads_per_block.y-1)/threads_per_block.y};

    CUDARawVolumeSample<<<blocks_per_grid,threads_per_block>>>(cu_sample_result,volume_texture);
    CUDA_RUNTIME_CHECK

    CUDA_RUNTIME_API_CALL(cudaMemcpy(data,cu_sample_result,(size_t)w*h,cudaMemcpyDeviceToHost));

    spdlog::info("Finish CUDA raw volume sample.");
}

__global__ void CUDARawVolumeSample(uint8_t *image, cudaTextureObject_t volume_data) {
    int image_x=blockIdx.x*blockDim.x+threadIdx.x;
    int image_y=blockIdx.y*blockDim.y+threadIdx.y;
    if(image_x>=sampleParameter.image_w || image_y>=sampleParameter.image_h) return;
    uint64_t image_idx=(uint64_t)image_y*sampleParameter.image_w+image_x;

    float3 virtual_sample_pos=sampleParameter.origin+(image_x-(int)sampleParameter.image_w/2)*sampleParameter.voxels_per_pixel.x*sampleParameter.right
                                                    +(image_y-(int)sampleParameter.image_h/2)*sampleParameter.voxels_per_pixel.y*sampleParameter.down;



    float3 physical_sample_pos=virtual_sample_pos*0.01f / sampleParameter.volume_board;


    image[image_idx]=tex3D<float>(volume_data,physical_sample_pos.x,physical_sample_pos.y,physical_sample_pos.z)*255;
//    image[image_idx]=255;
}


VS_END
