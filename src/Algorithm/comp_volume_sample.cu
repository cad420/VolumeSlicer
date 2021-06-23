#include "Algorithm/comp_volume_sample.cuh"
#include "Algorithm/helper_math.h"
VS_START
__constant__ CompSampleParameter compSampleParameter;
__constant__ BlockParameter blockParameter;
//__constant__ variable's size should be known when compile
uint4* d_mappingTable=nullptr;
__constant__ uint4* mappingTable;
__constant__ uint lodMappingTableOffset[10];//!max lod is 9 and offset is for uint4 not uint
__constant__ cudaTextureObject_t* cacheVolumes;

__device__ int PowII(int x,int y){
    int res=1;
    for(int i=0;i<y;i++){
        res*=x;
    }
    return res;
}

//sample_pos must in voxel world-coord
__device__ float VirtualSample(float3 sample_pos){
    int lod_t=PowII(2,compSampleParameter.lod);
    int lod_no_padding_block_length=blockParameter.no_padding_block_length*lod_t;
    int3 virtual_block_index=make_int3(sample_pos / lod_no_padding_block_length);
    int3 lod_block_dim=make_int3(make_float3(blockParameter.block_dim + lod_t-1) / lod_t);
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

    if(physical_block_flag==0){
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

    //todo
    float3 virtual_sample_pos;

    image[image_idx]=VirtualSample(virtual_sample_pos)*255;

}

void SetCUDASampleParameter(const CompSampleParameter& parameter){
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(compSampleParameter,&parameter,sizeof(CompSampleParameter)));
}
void SetCUDABlockParameter(const BlockParameter& parameter){
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(blockParameter,&parameter,sizeof(BlockParameter)));
}

//size is element num for data
void CreateCUDAMappingTable(uint32_t * data,size_t size){
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_mappingTable,size*sizeof(uint32_t)));
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(mappingTable,&d_mappingTable,sizeof(uint4*)));
    CUDA_RUNTIME_API_CALL(cudaMemcpy(d_mappingTable,data,size*sizeof(uint32_t),cudaMemcpyHostToDevice));
}

//size is element num for data
void UpdateCUDAMappingTable(uint32_t * data,size_t size){
    CUDA_RUNTIME_API_CALL(cudaMemcpy(d_mappingTable,data,size*sizeof(uint32_t),cudaMemcpyHostToDevice));
}

//max size is 10
void SetCUDAMappingTableOffset(uint32_t* data,size_t size){
        CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(lodMappingTableOffset,data,size*sizeof(uint32_t)));
}

//only create uint8_t 3D CUDA Texture
void CreateCUDATexture3D(cudaExtent textureSize, cudaArray **ppCudaArray, cudaTextureObject_t *pCudaTextureObject){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
    CUDA_RUNTIME_API_CALL(cudaMalloc3DArray(ppCudaArray, &channelDesc, textureSize));
    cudaResourceDesc            texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = *ppCudaArray;
    cudaTextureDesc             texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = true; // access with normalized texture coordinates
    texDesc.filterMode       = cudaFilterModeLinear; // linear interpolation
    texDesc.borderColor[0]=0.f;
    texDesc.borderColor[1]=0.f;
    texDesc.borderColor[2]=0.f;
    texDesc.borderColor[3]=0.f;
    texDesc.addressMode[0]=cudaAddressModeBorder;
    texDesc.addressMode[1]=cudaAddressModeBorder;
    texDesc.addressMode[2]=cudaAddressModeBorder;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    CUDA_RUNTIME_API_CALL(cudaCreateTextureObject(pCudaTextureObject, &texRes, &texDesc, NULL));

}

//offset if count by byte
void UpdateCUDATexture3D(uint8_t* data,cudaArray* pCudaArray,uint32_t block_length,uint32_t x_offset,uint32_t y_offset,uint32_t z_offset){
    CUDA_MEMCPY3D m={0};
    m.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    m.srcDevice=(CUdeviceptr)data;

    m.dstMemoryType=CU_MEMORYTYPE_ARRAY;
    m.dstArray=(CUarray)pCudaArray;
    m.dstXInBytes=x_offset;
    m.dstY=y_offset;
    m.dstZ=z_offset;

    m.WidthInBytes=block_length;
    m.Height=block_length;
    m.Depth=block_length;

    CUDA_DRIVER_API_CALL(cuMemcpy3D(&m));
}


bool CUDACompVolumeSampler::getTexturePos(const std::array<uint32_t, 4> &target, std::array<uint32_t, 4> &pos) {
    for(const auto& it:block_table){
        if(it.block_index==target && it.cached){
            assert(!it.valid);
            spdlog::info("Copy CUDA device memory to CUDA Array which already stored.");
            pos=it.pos_index;
            return true;
        }
    }
    for(const auto& it:block_table){
        if(!it.valid && !it.cached){
            pos=it.pos_index;
            return false;
        }
    }
    for(const auto& it:block_table){
        if(!it.valid){
            pos=it.pos_index;
            return false;
        }
    }
    throw std::runtime_error("Can't find empty pos in CUDA Textures");
}

bool CUDACompVolumeSampler::IsCachedBlock(const std::array<uint32_t, 4> &target) const {
    for(const auto& it:block_table){
        if(it.block_index==target && it.cached){
            return true;
        }
    }
    return false;
}

bool CUDACompVolumeSampler::SetCachedBlockValid(const std::array<uint32_t, 4> &target) {
    for(auto& it:block_table){
        if(it.block_index==target && it.cached){
            it.valid=true;
            //!if find then should update mapping_table
            updateMappingTable(target,it.pos_index);

            return true;
        }
    }
    return false;
}
void CUDACompVolumeSampler::SetBlockInvalid(const std::array<uint32_t, 4> &target) {
    for(auto& it:block_table){
        if(it.block_index==target){
            it.valid=false;
        }
    }
}
void CUDACompVolumeSampler::createBlockTable() {
    for(uint32_t t=0;t<cu_array_num;t++){
        for(uint32_t k=0;k<cu_array_size[2]/block_length;k++){
            for(uint32_t j=0;j<cu_array_size[1]/block_length;j++){
                for(uint32_t i=0;i<cu_array_size[0]/block_length;i++){
                    BlockTableItem item;
                    item.pos_index={i,j,k,t};
                    item.block_index={INVALID,INVALID,INVALID,INVALID};
                    item.valid=false;
                    item.cached=false;
                    block_table.push_back(item);
                }
            }
        }
    }
}

void CUDACompVolumeSampler::CreateMappingTable(const std::map<uint32_t, std::array<uint32_t, 3>> &lod_block_dim) {
    this->lod_block_dim=lod_block_dim;
    lod_mapping_table_offset[lod_block_dim.begin()->first]=0;
    this->min_lod=0xffffffff;
    this->max_lod=0;
    for(auto it=lod_block_dim.begin();it!=lod_block_dim.end();it++){
        this->min_lod=it->first<min_lod?it->first:min_lod;
        this->max_lod=it->first>max_lod?it->first:max_lod;
        auto & t=it->second;
        size_t lod_block_num=(size_t)t[0]*t[1]*t[2];
        lod_mapping_table_offset[it->first+1]=lod_mapping_table_offset[it->first]+lod_block_num*4;
    }
    mapping_table.assign(lod_mapping_table_offset.at(max_lod+1),0);
    CreateCUDAMappingTable(mapping_table.data(),mapping_table.size());
    uint32_t offset[10];
    memset(offset,0,10*sizeof(uint32_t));
    for(uint32_t i=min_lod;i<=max_lod;i++){
        offset[i]=lod_mapping_table_offset.at(i)/4;
    }
    SetCUDAMappingTableOffset(offset,10);
}

void CUDACompVolumeSampler::UploadMappingTable() {
    UpdateCUDAMappingTable(mapping_table.data(),mapping_table.size());
}

void CUDACompVolumeSampler::UploadCompSampleParameter(const CompSampleParameter &sampleParameter) {
    SetCUDASampleParameter(sampleParameter);
}

void CUDACompVolumeSampler::UploadBlockParameter(const BlockParameter & blockParameter) {
    SetCUDABlockParameter(blockParameter);
}

void CUDACompVolumeSampler::SetCUDATextures(uint32_t tex_num, uint32_t tex_x, uint32_t tex_y, uint32_t tex_z) {
    auto tex_size=make_cudaExtent(tex_x,tex_y,tex_z);
    this->cu_array_num=tex_num;
    this->cu_array_size={tex_x,tex_y,tex_z};
    cu_arrays.resize(tex_num,nullptr);
    cache_volumes.resize(tex_num,0);
    for(uint32_t i=0;i<cu_array_num;i++){
        CreateCUDATexture3D(tex_size,&cu_arrays[i],&cache_volumes[i]);
    }
    createBlockTable();
}

void CUDACompVolumeSampler::UploadCUDATexture3D(const std::array<uint32_t, 4> &index, uint8_t *data, size_t size) {
    std::array<uint32_t,4> pos;
    bool cached=getTexturePos(index,pos);
    if(!cached){

        UpdateCUDATexture3D(data,cu_arrays[pos[3]],block_length,block_length*pos[0],block_length*pos[1],block_length*pos[2]);
        spdlog::info("Upload block({0},{1},{2},{3}) to CUDA Array({4},{5},{6},{7})",
                     index[0],index[1],index[2],index[3],
                     pos[0],pos[1],pos[2],pos[3]);
    }
    else
        spdlog::info("UploadCUDATexture3D which has already been cached.");

    for(auto& it:block_table){
        if(it.pos_index==pos){
            if(cached){
                assert(it.cached && it.block_index==index);
            }
            if(it.block_index!=index && it.block_index[0]!=INVALID){
                updateMappingTable(it.block_index,{0,0,0,0},false);
            }
            it.block_index=index;
            it.valid=true;
            it.cached=true;
        }
    }

    updateMappingTable(index,pos);
    UploadMappingTable();
}

void CUDACompVolumeSampler::updateMappingTable(const std::array<uint32_t, 4> &index,const std::array<uint32_t,4>& pos,bool valid) {
    size_t flat_idx=(index[2]*lod_block_dim.at(index[3])[0]*lod_block_dim.at(index[3])[1]
                     +index[1]*lod_block_dim.at(index[3])[0]
                     +index[0])*4+lod_mapping_table_offset.at(index[3]);
    mapping_table[flat_idx+0]=pos[0];
    mapping_table[flat_idx+1]=pos[1];
    mapping_table[flat_idx+2]=pos[2];
    if(valid)
        mapping_table[flat_idx+3]=pos[3]|(0x00010000);
    else
        mapping_table[flat_idx+3]&=0x0000ffff;
}

void CUDACompVolumeSampler::Sample(uint8_t *data, Slice slice, float space_x, float space_y, float space_z) {
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
    spdlog::info("Finish CUDA comp volume sample.");
}

void CUDACompVolumeSampler::SetBlockInfo(uint32_t block_length,uint32_t padding) {
    this->block_length=block_length;
    this->padding=padding;
}


VS_END

