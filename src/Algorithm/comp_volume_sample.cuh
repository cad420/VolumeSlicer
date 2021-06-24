#pragma once

#include<VolumeSlicer/helper.hpp>
#include<VolumeSlicer/slice.hpp>
#include<list>
#include<map>
VS_START

struct CompSampleParameter{
    int image_w;
    int image_h;
    int lod;
    float3 volume_board;//dim*space
    float2 voxels_per_pixel;
    float3 origin;
    float3 right;
    float3 down;
    float3 space;
};

struct BlockParameter{
    int block_length;
    int padding;
    int no_padding_block_length;
    int3 block_dim;//for lod 0
    int3 texture_size3;//a single texture's dim

};



/**
 * main task is managing blocks in CUDA textures, like block data should copy to where
 * todo may inherit from an ISampler to hide using CUDA or OpenGL or even CPU
 */
class CUDACompVolumeSampler{
public:
    CUDACompVolumeSampler()
    :old_h(0),old_w(0),cu_sample_result(nullptr)
    {
        CUDA_DRIVER_API_CALL(cuInit(0));
        int cu_device_cnt=0;
        CUdevice cu_device;
        int using_gpu=0;
        char using_device_name[80];
        CUDA_DRIVER_API_CALL(cuDeviceGetCount(&cu_device_cnt));
        CUDA_DRIVER_API_CALL(cuDeviceGet(&cu_device,using_gpu));
        CUDA_DRIVER_API_CALL(cuDeviceGetName(using_device_name,sizeof(using_device_name),cu_device));
        this->cu_ctx=nullptr;
        CUDA_DRIVER_API_CALL(cuCtxCreate(&cu_ctx,0,cu_device));
    }

    void SetBlockInfo(uint32_t block_length,uint32_t padding);

    void Sample(uint8_t* data,Slice slice,float space_x,float space_y,float space_z);

    bool IsCachedBlock(const std::array<uint32_t,4>&) const;

    //if target block is cached, then set it valid and return true
    //!and will update mapping_table
    //if target block not found in cached blocks, return false.
    bool SetCachedBlockValid(const std::array<uint32_t,4>&);

    void SetBlockInvalid(const std::array<uint32_t,4>&);

    void UploadCompSampleParameter(const CompSampleParameter&);

    void UploadBlockParameter(const BlockParameter&);

    //copy device data to CUDA array
    //and will update mapping_table if copy successfully
    void UploadCUDATexture3D(const std::array<uint32_t,4>&,uint8_t*,size_t);

    //create cuda array and texture obj
    void SetCUDATextures(uint32_t tex_num, uint32_t tex_x, uint32_t tex_y, uint32_t tex_z);

    void CreateMappingTable(const std::map<uint32_t,std::array<uint32_t,3>>&);

private:
    struct BlockTableItem{
        std::array<uint32_t,4> block_index;
        std::array<uint32_t,4> pos_index;
        bool valid;
        bool cached;
    };
private:
    void uploadCUDATextureObject();

    void updateMappingTable(const std::array<uint32_t,4>& index,const std::array<uint32_t,4>& pos,bool valid=true);

    void createBlockTable();

    //call after every UploadCUDATexture3D
    void uploadMappingTable();

    //get empty or valid pos in one CUDA texture
    bool getTexturePos(const std::array<uint32_t,4>&,std::array<uint32_t,4>&);
private:
    CUcontext cu_ctx;

    uint32_t block_length,padding;

    int old_w,old_h;
    uint8_t* cu_sample_result;//image

    uint32_t cu_array_num;
    std::array<uint32_t,3> cu_array_size;
    std::vector<cudaArray*> cu_arrays;
    std::vector<cudaTextureObject_t> cache_volumes;

    std::list<BlockTableItem> block_table;

    uint32_t min_lod,max_lod;
    std::vector<uint32_t> mapping_table;
    std::map<uint32_t,std::array<uint32_t,3>> lod_block_dim;
    //!offset is for uint
    std::unordered_map<uint32_t,uint32_t> lod_mapping_table_offset;
};





VS_END