//
// Created by wyz on 2021/9/13.
//
#include "cuda_offscreen_comp_render_impl.cuh"

#include "Common/cuda_box.hpp"
#include "Common/cuda_utils.hpp"

#include <VolumeSlicer/frame.hpp>
#include <VolumeSlicer/vec.hpp>
#include <VolumeSlicer/color.hpp>
#include <stdgpu/cstddef.h>
#include <stdgpu/unordered_set.cuh>
#include <thrust/host_vector.h>
using namespace CUDAOffRenderer;

namespace {

__constant__ CUDAOffCompRenderParameter cudaOffCompRenderParameter;
__constant__ CompVolumeParameter        compVolumeParameter;
__constant__ ShadingParameter           shadingParameter;
__constant__ float3* ray_directions;//Image<Vec4> is not valid in cuda kernel
__constant__ float3* ray_start_pos;
__constant__ float3* ray_stop_pos;
__constant__ float4* intermediate_result;
__constant__ uint*   color_image;
__constant__ uint4*  mapping_table;
__constant__ uint    lod_mapping_table_offset[10];
__constant__ cudaTextureObject_t cache_volumes[20];
__constant__ cudaTextureObject_t transfer_func;
__constant__ cudaTextureObject_t preInt_transfer_func;
float3* d_ray_directions      = nullptr;
float3* d_ray_start_pos       = nullptr;
float3* d_ray_stop_pos        = nullptr;
float4* d_intermediate_result = nullptr;
uint*   d_color_image         = nullptr;
uint4*  d_mapping_table       = nullptr;
int     d_image_w             = 0;
int     d_image_h             = 0;
cudaArray* tf                 = nullptr;
cudaArray* preInt_tf          = nullptr;





__device__ int VirtualSample(int lod,int lod_t,const float3& sample_pos,float& scalar,
                             stdgpu::unordered_set<int4,Hash_Int4>& missed_blocks);

__device__ float3 PhongShading(int lod,int lod_t,stdgpu::unordered_set<int4,Hash_Int4>& missed_blocks,
                               float3 const& sample_pos,
                               float3 const& diffuse_color,
                               float3 const& view_direction);

__device__ uint RGBAFloatToUInt(float4 rgba){
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void CUDAGenRays(){
    int image_x   = blockIdx.x*blockDim.x+threadIdx.x;
    int image_y   = blockIdx.y*blockDim.y+threadIdx.y;
    int image_idx = image_y * cudaOffCompRenderParameter.image_w + image_x;
    if(image_x>=cudaOffCompRenderParameter.image_w || image_y>=cudaOffCompRenderParameter.image_h) return;

    double scale = 2.f * tan(radians(cudaOffCompRenderParameter.fov / 2.f)) / cudaOffCompRenderParameter.image_h;
    double ratio = 1.f *cudaOffCompRenderParameter.image_w / cudaOffCompRenderParameter.image_h;
    double x     = (image_x + 0.5f - cudaOffCompRenderParameter.image_w / 2.f) * scale * ratio;
    double y     = (cudaOffCompRenderParameter.image_h / 2.f - image_y - 0.5f) * scale;

    float3 pixel_view_pos = cudaOffCompRenderParameter.camera_pos
                          + cudaOffCompRenderParameter.front
                          + cudaOffCompRenderParameter.right * x
                          + cudaOffCompRenderParameter.up * y;
    float3 pixel_view_direction = normalize(pixel_view_pos-cudaOffCompRenderParameter.camera_pos);
    ray_directions[image_idx]   =  pixel_view_direction;
//    color_image[image_idx]      = RGBAFloatToUInt(make_float4(pixel_view_direction,1.f));
//    return ;
    auto intersect_t=IntersectWithAABB(CUDABox(make_float3(0.f),compVolumeParameter.volume_board),
                                         CUDASimpleRay(cudaOffCompRenderParameter.camera_pos,pixel_view_direction));

    if(IsIntersected(intersect_t.x,intersect_t.y)){
        if(intersect_t.x>0.f){
            ray_start_pos[image_idx]=cudaOffCompRenderParameter.camera_pos+intersect_t.x*pixel_view_direction;
        }
        else{
            ray_start_pos[image_idx]=cudaOffCompRenderParameter.camera_pos;
        }
        ray_stop_pos[image_idx]=cudaOffCompRenderParameter.camera_pos+intersect_t.y*pixel_view_direction;
    }
    else{
        ray_stop_pos[image_idx]=ray_start_pos[image_idx]=make_float3(0.f);
    }
//    color_image[image_idx] = RGBAFloatToUInt(make_float4(ray_stop_pos[image_idx]/compVolumeParameter.volume_board,1.f));

}

__device__ int VirtualSample(int lod,int lod_t,const float3& sample_pos,float& scalar,
                             stdgpu::unordered_set<int4,Hash_Int4>& missed_blocks){

    int3 virtual_block_idx=make_int3(sample_pos / compVolumeParameter.no_padding_block_length);
    if(virtual_block_idx.x < 0 || virtual_block_idx.y < 0 || virtual_block_idx.z < 0 ||
       virtual_block_idx.x >= compVolumeParameter.volume_dim.x ||
       virtual_block_idx.y >= compVolumeParameter.volume_dim.y ||
       virtual_block_idx.z >= compVolumeParameter.volume_dim.z ){
        scalar = 0.f;
        return 1;
    }
    virtual_block_idx = virtual_block_idx / lod_t;
    int3 block_dim=(compVolumeParameter.block_dim+lod_t-1)/lod_t;
    size_t flat_virtual_block_idx = virtual_block_idx.z * block_dim.x * block_dim.y
                                  + virtual_block_idx.y * block_dim.x
                                  + virtual_block_idx.x+ lod_mapping_table_offset[lod];

    uint4 physical_block_idx  = mapping_table[flat_virtual_block_idx];

    uint  physical_block_flag = (physical_block_idx.w>>16)&(0x0000ffff);
    if(physical_block_flag==0){
        missed_blocks.insert(make_int4(virtual_block_idx,lod));
        scalar = 0.f;
        return 0;
    }
    uint physical_texture_idx = physical_block_idx.w & 0x0000ffff;
    float3 offset_in_no_padding_block = (sample_pos-make_float3(virtual_block_idx* compVolumeParameter.no_padding_block_length*lod_t))/lod_t;
    float3 physical_sample_pos = make_float3(physical_block_idx.x,physical_block_idx.y,physical_block_idx.z)* compVolumeParameter.block_length
                                 + offset_in_no_padding_block + make_float3(compVolumeParameter.padding);
    physical_sample_pos /= make_float3(compVolumeParameter.volume_texture_shape);
    scalar = tex3D<float>(cache_volumes[physical_texture_idx],physical_sample_pos.x,physical_sample_pos.y,physical_sample_pos.z);
    return 1;

}

__device__ int EvaluateLod(float distance){
//    return 0;
    if(distance<0.3f){
        return 0;
    }
    else if(distance<0.5f){
        return 1;
    }
    else if(distance<1.2f){
        return 2;
    }
    else if(distance<1.6f){
        return 3;
    }
    else if(distance<3.2f){
        return 4;
    }
    else if(distance<6.4f){
        return 5;
    }
    else{
        return 6;
    }
}

__device__ int IntPow(int x,int y){
    int ans=1;
    for(int i=0;i<y;i++)
        ans *= x;
    return ans;
}

__global__ void CUDARenderPass(stdgpu::unordered_set<int4,Hash_Int4> missed_blocks){
    int image_x   = blockIdx.x*blockDim.x+threadIdx.x;
    int image_y   = blockIdx.y*blockDim.y+threadIdx.y;
    int image_idx = image_y * cudaOffCompRenderParameter.image_w + image_x;

    if(image_x>=cudaOffCompRenderParameter.image_w || image_y>=cudaOffCompRenderParameter.image_h) return;
//    missed_blocks.insert(make_int4(image_x,image_y,0,0));
//    return;
    float4 color = intermediate_result[image_idx];
    if(color.w > 0.99f) return;
    float3 last_ray_start_pos = ray_start_pos[image_idx];
    float3 last_ray_stop_pos  = ray_stop_pos[image_idx];
    float3 ray_direction      = last_ray_stop_pos - last_ray_start_pos;


    float3 ray_sample_pos     = last_ray_start_pos;
    float sample_scalar;
    int i = 0;
    int lod_steps = 0;
    int last_lod  = EvaluateLod(length(ray_sample_pos-cudaOffCompRenderParameter.camera_pos));
    int last_lod_t = IntPow(2,last_lod);
    int steps      = length(ray_direction)/cudaOffCompRenderParameter.step/last_lod_t;
    ray_direction             = normalize(ray_direction);
    float3 lod_sample_start_pos = last_ray_start_pos;
    for(;i<steps;i++){
        int cur_lod=EvaluateLod(length(ray_sample_pos-cudaOffCompRenderParameter.camera_pos));
        int lod_t = IntPow(2,cur_lod);
        if(cur_lod > last_lod){
            lod_sample_start_pos=ray_sample_pos;
            last_lod = cur_lod;
            lod_steps = i;
        }

        int flag = VirtualSample(cur_lod,lod_t,ray_sample_pos/compVolumeParameter.volume_space,sample_scalar,missed_blocks);
        if(flag == 0 ){
            ray_start_pos[image_idx]       = ray_sample_pos;
            intermediate_result[image_idx] = color;
            return;
        }

        if(sample_scalar > 0.f){
            float4 sample_color = tex1D<float4>(transfer_func,sample_scalar);
            if(sample_color.w > 0.f){
                auto shading_color = make_float4(PhongShading(cur_lod,lod_t,missed_blocks,
                                                  ray_sample_pos / compVolumeParameter.volume_space,
                                                  make_float3(sample_color),
                                                  ray_direction),sample_color.w);
                color = color + shading_color* make_float4(sample_color.w, sample_color.w, sample_color.w, 1.f) * (1.f - color.w);
                if(color.w > 0.99f){
                    break;
                }
            }
        }

        ray_sample_pos = lod_sample_start_pos+ (i+1-lod_steps)*ray_direction * cudaOffCompRenderParameter.step * lod_t;
    }
    if(i >= steps || color.w > 0.99f){
        color.w=1.f;
        intermediate_result[image_idx] = color;
        color_image[image_idx]         = RGBAFloatToUInt(color);
//        color_image[image_idx]         = RGBAFloatToUInt(make_float4(ray_sample_pos/compVolumeParameter.volume_board,1.f));
    }
}

__device__ float3 PhongShading(int lod,int lod_t,stdgpu::unordered_set<int4,Hash_Int4>& missed_blocks,
                               float3 const& sample_pos,
                               float3 const& diffuse_color,
                               float3 const& view_direction){
    float3 N=make_float3(0.f);
    float x1,x2;
    VirtualSample(lod,lod_t,sample_pos+make_float3( compVolumeParameter.voxel*lod_t,0.f,0.f),x1,missed_blocks);
    VirtualSample(lod,lod_t,sample_pos+make_float3(-compVolumeParameter.voxel*lod_t,0.f,0.f),x2,missed_blocks);
    N.x = x1 - x2;
    VirtualSample(lod,lod_t,sample_pos+make_float3(0.f, compVolumeParameter.voxel*lod_t,0.f),x1,missed_blocks);
    VirtualSample(lod,lod_t,sample_pos+make_float3(0.f,-compVolumeParameter.voxel*lod_t,0.f),x2,missed_blocks);
    N.y = x1 - x2;
    VirtualSample(lod,lod_t,sample_pos+make_float3(0.f,0.f, compVolumeParameter.voxel*lod_t),x1,missed_blocks);
    VirtualSample(lod,lod_t,sample_pos+make_float3(0.f,0.f,-compVolumeParameter.voxel*lod_t),x2,missed_blocks);
    N.z = x1 - x2;
    N = normalize(-N);

    float3 L = make_float3(0.f) - view_direction;
    float3 R = L;

    float3 ambient=shadingParameter.ka*diffuse_color;
    float3 specular=shadingParameter.ks*pow(max(dot(N,(L+R)/2.f),0.f),shadingParameter.shininess)*make_float3(1.f);
    float3 diffuse=shadingParameter.kd*max(dot(N,L),0.f)*diffuse_color;
    return ambient+specular+diffuse;
}
void CreateDeviceRenderImages(int w,int h){
    d_image_w = w;
    d_image_h = h;

    {
        if(d_ray_directions){
            CUDA_RUNTIME_API_CALL(cudaFree(d_ray_directions));
        }
        CUDA_RUNTIME_API_CALL(cudaMalloc(&d_ray_directions,sizeof(float3)*d_image_w*d_image_h));
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(ray_directions,&d_ray_directions,sizeof(float3*)));
    }
    {
        if(d_ray_start_pos){
            CUDA_RUNTIME_API_CALL(cudaFree(d_ray_start_pos));
        }
        CUDA_RUNTIME_API_CALL(cudaMalloc(&d_ray_start_pos,sizeof(float3)*d_image_w*d_image_h));
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(ray_start_pos,&d_ray_start_pos,sizeof(float3*)));
    }
    {
        if(d_ray_stop_pos){
            CUDA_RUNTIME_API_CALL(cudaFree(d_ray_stop_pos));
        }
        CUDA_RUNTIME_API_CALL(cudaMalloc(&d_ray_stop_pos,sizeof(float3)*d_image_w*d_image_h));
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(ray_stop_pos,&d_ray_stop_pos,sizeof(float3*)));
    }
    {
        if(d_intermediate_result){
            CUDA_RUNTIME_API_CALL(cudaFree(d_intermediate_result));
        }
        CUDA_RUNTIME_API_CALL(cudaMalloc(&d_intermediate_result,sizeof(float4)*d_image_w*d_image_h));
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(intermediate_result,&d_intermediate_result,sizeof(float4*)));
    }
    {
        if(d_color_image){
            CUDA_RUNTIME_API_CALL(cudaFree(d_color_image));
        }
        CUDA_RUNTIME_API_CALL(cudaMalloc(&d_color_image,sizeof(uint)*d_image_w*d_image_h));
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(color_image,&d_color_image,sizeof(uint*)));
    }

}

void CreateDeviceMappingTable(const uint32_t*data,size_t size){
    if(d_mapping_table){
        CUDA_RUNTIME_API_CALL(cudaFree(d_mapping_table));
    }
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_mapping_table,sizeof(uint32_t)*size));
    assert(d_mapping_table);
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(mapping_table,&d_mapping_table,sizeof(uint4*)));

}

}



namespace CUDAOffRenderer{

    void UploadTransferFunc(float *data, size_t size)
    {
        cudaTextureObject_t tf_tex;
        if(!tf){
            CreateCUDATexture1D(256,&tf,&tf_tex);
        }
        UpdateCUDATexture1D(reinterpret_cast<uint8_t*>(data),tf,sizeof(float)*256*4,0);
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(transfer_func,&tf_tex,sizeof(cudaTextureObject_t)));
    }

    void UploadPreIntTransferFunc(float *data, size_t size)
    {
        cudaTextureObject_t pre_tf_tex;
        if(!preInt_tf){
            CreateCUDATexture2D(256,256,&preInt_tf,&pre_tf_tex);
        }
        UpdateCUDATexture2D(reinterpret_cast<uint8_t*>(data),preInt_tf,sizeof(float)*256*4,256,0,0);
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(preInt_transfer_func,&pre_tf_tex,sizeof(cudaTextureObject_t)));
    }

    void UploadCUDAOffCompRenderParameter(const CUDAOffCompRenderParameter & comp_render)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cudaOffCompRenderParameter,&comp_render,sizeof(CUDAOffCompRenderParameter)));
    }

    void UploadCompVolumeParameter(const CompVolumeParameter & comp_volume)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(compVolumeParameter,&comp_volume,sizeof(CompVolumeParameter)));
    }

    void UploadShadingParameter(const ShadingParameter & shading)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(shadingParameter,&shading,sizeof(ShadingParameter)));
    }

    void UploadMappingTable(const uint32_t *data, size_t size)
    {
        if(!d_mapping_table){
            CreateDeviceMappingTable(data,size);
        }
        CUDA_RUNTIME_API_CALL(cudaMemcpy(d_mapping_table,data,sizeof(uint32_t)*size,cudaMemcpyHostToDevice));
    }

    void UploadLodMappingTableOffset(const uint32_t *data, size_t size)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(lod_mapping_table_offset,data,sizeof(uint32_t)*size));
    }

    void SetCUDATextureObject(cudaTextureObject_t *textures, size_t size)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cache_volumes,textures,sizeof(cudaTextureObject_t)*size));
    }



    void CUDARenderPrepare(int w,int h)
    {
        if(d_image_w != w || d_image_h != h){
            CreateDeviceRenderImages(w,h);
        }
        dim3 threads_per_block={16,16};
        dim3 blocks_per_grid={(d_image_w+threads_per_block.x-1)/threads_per_block.x,(d_image_h+threads_per_block.y-1)/threads_per_block.y};

        CUDAGenRays<<<blocks_per_grid,threads_per_block>>>();
        CUDA_RUNTIME_CHECK


    }

    void CUDARender(std::unordered_set<int4,Hash_Int4>& h_missed_blocks)
    {
        dim3 threads_per_block={16,16};
        dim3 blocks_per_grid={(d_image_w+threads_per_block.x-1)/threads_per_block.x,(d_image_h+threads_per_block.y-1)/threads_per_block.y};

        stdgpu::unordered_set<int4,Hash_Int4> d_missed_blocks = stdgpu::unordered_set<int4,Hash_Int4>::createDeviceObject(4096);

        CUDARenderPass<<<blocks_per_grid,threads_per_block>>>(d_missed_blocks);
        CUDA_RUNTIME_CHECK

        assert(h_missed_blocks.empty());
        auto d=d_missed_blocks.device_range();
        thrust::host_vector<int4> v(d.size());
        thrust::copy(d.begin(),d.end(),v.begin());
        for(auto& block:v){
            h_missed_blocks.insert(block);
        }
    }
    void GetRenderImage(uint8_t *data)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpy(data,d_color_image,sizeof(uint)*d_image_w*d_image_h,cudaMemcpyDefault));
    }
}