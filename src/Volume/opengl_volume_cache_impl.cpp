//
// Created by wyz on 2021/10/20.
//
#include "opengl_volume_cache_impl.hpp"
#include <glad/glad.h>
#include <Utils/logger.hpp>
#include <iostream>
#include <cudaGL.h>
#include <Common/gl_helper.hpp>
VS_START
std::unique_ptr<OpenGLVolumeBlockCache> vs::OpenGLVolumeBlockCache::Create()
{
    return std::make_unique<OpenGLVolumeBlockCacheImpl>();
}
OpenGLVolumeBlockCacheImpl::OpenGLVolumeBlockCacheImpl()
{
    //todo 0 replace with iGPU
    SetCUDACtx(0);
    LOG_INFO("Create OpenGLVolumeBlockCache.");
}
OpenGLVolumeBlockCacheImpl::~OpenGLVolumeBlockCacheImpl()
{
    LOG_INFO("Call ~OpenGLVolumeBlockCacheImpl destructor.");
    for(auto& rc:cu_resources){
        CUDA_DRIVER_API_CALL(cuGraphicsUnregisterResource(rc));
    }
    glDeleteTextures(gl_textures.size(),gl_textures.data());
}
void OpenGLVolumeBlockCacheImpl::SetCacheBlockLength(uint32_t block_length)
{
    this->block_length = block_length;
}
void OpenGLVolumeBlockCacheImpl::SetCacheCapacity(uint32_t num, uint32_t x, uint32_t y, uint32_t z)
{
    this->gl_tex_num = num;
    this->gl_tex_shape = {x,y,z};
    this->gl_textures.resize(num,0);
    this->cu_resources.resize(num,nullptr);
    GL_EXPR(glCreateTextures(GL_TEXTURE_3D,num,gl_textures.data()));
    for(auto& tex:gl_textures){
        GL_EXPR(glTextureStorage3D(tex,1,GL_R8,x,y,z));
    }
    for(int i = 0 ;i<num;i++){
        CUDA_DRIVER_API_CALL(cuGraphicsGLRegisterImage(&cu_resources[i],gl_textures[i],
                                                       GL_TEXTURE_3D,CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    }
    this->createBlockCacheTable();
    LOG_INFO("SetCacheCapacity, num:{0} x:{1} y:{2} z:{3}.",num,x,y,z);
}
auto OpenGLVolumeBlockCacheImpl::GetCacheShape() -> std::array<uint32_t, 4>
{
    return std::array<uint32_t ,4>{gl_tex_num,gl_tex_shape[0],gl_tex_shape[1],gl_tex_shape[2]};
}
void OpenGLVolumeBlockCacheImpl::CreateMappingTable(const std::map<uint32_t, std::array<uint32_t, 3>> &lod_block_dim)
{
    this->lod_block_dim=lod_block_dim;
    this->lod_mapping_table_offset[lod_block_dim.begin()->first]=0;
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

}
void OpenGLVolumeBlockCacheImpl::UploadVolumeBlock(const std::array<uint32_t, 4> &index, uint8_t *data, size_t size,bool device)
{
    //upload data to texture
    std::array<uint32_t ,4> pos{INVALID,INVALID,INVALID,INVALID};
    bool cached=getCachedPos(index,pos);
    if(!cached){
        //todo
        CUDA_DRIVER_API_CALL(cuGraphicsMapResources(1,&cu_resources[pos[3]],0));
        CUarray cu_array;
        CUDA_DRIVER_API_CALL(cuGraphicsSubResourceGetMappedArray(&cu_array,cu_resources[pos[3]],0,0));
        CUDA_MEMCPY3D m = {0};
        if(device){
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            m.srcDevice = (CUdeviceptr)data;
        }
        else{
            m.srcMemoryType=CU_MEMORYTYPE_HOST;
            m.srcHost=data;
        }

        m.dstMemoryType=CU_MEMORYTYPE_ARRAY;
        m.dstArray = cu_array;
        m.dstXInBytes = pos[0]*block_length;
        m.dstY = pos[1]*block_length;
        m.dstZ = pos[2]*block_length;

        m.WidthInBytes = block_length;
        m.Height = block_length;
        m.Depth = block_length;
        CUDA_DRIVER_API_CALL(cuMemcpy3D(&m));
        CUDA_DRIVER_API_CALL(cuGraphicsUnmapResources(1,&cu_resources[pos[3]],0));
        LOG_INFO("Upload block({0},{1},{2},{3}) to OpenGL texture({4},{5},{6},{7})",
                     index[0],index[1],index[2],index[3],
                     pos[0],pos[1],pos[2],pos[3]);
    }
    else{
        spdlog::info("UploadVolumeBlock which has already been cached.");
    }
    //update block_cache_table
    for(auto& it:block_cache_table){
        if(it.pos_index == pos){
            if(cached){
                assert(it.cached && it.block_index == index);
            }
            if(it.block_index!=index && it.block_index[0]!=INVALID){
                this->updateMappingTable(it.block_index,{0,0,0,0},false);
            }
            it.block_index=index;
            it.valid=true;
            it.cached=true;
        }
    }

    //update mapping_table
    updateMappingTable(index,pos);
}
bool OpenGLVolumeBlockCacheImpl::IsCachedBlock(const std::array<uint32_t, 4> &target)
{
    for(auto& it:block_cache_table){
        if(it.block_index == target){
            return it.cached;
        }
    }
}
bool OpenGLVolumeBlockCacheImpl::IsValidBlock(const std::array<uint32_t, 4> &target)
{
    for(auto& it:block_cache_table){
        if(it.block_index == target){
            return it.valid;
        }
    }
}
auto OpenGLVolumeBlockCacheImpl::GetBlockStatus(const std::array<uint32_t, 4> &target) -> std::array<bool, 2>
{
    for(auto& it:block_cache_table){
        if(it.block_index == target){
            return {it.valid,it.cached};
        }
    }
}
int OpenGLVolumeBlockCacheImpl::GetRemainEmptyBlock() const
{
    int cnt=0;
    for(auto& it:block_cache_table){
        if(!it.valid)
            cnt++;
    }
    return cnt;
}
void OpenGLVolumeBlockCacheImpl::clear()
{
    for(auto& it:block_cache_table){
        it.valid=false;
    }
    mapping_table.assign(mapping_table.size(),0);
}
bool OpenGLVolumeBlockCacheImpl::SetCachedBlockValid(const std::array<uint32_t, 4> &target)
{
    for(auto& it:block_cache_table){
        if(it.block_index==target && it.cached){
            it.valid=true;
            this->updateMappingTable(target,it.pos_index,true);
            return true;
        }
    }
    return false;
}
void OpenGLVolumeBlockCacheImpl::SetBlockInvalid(const std::array<uint32_t, 4> &target)
{
    for(auto& it:block_cache_table){
        if(it.block_index ==  target){
            it.valid=false;
            return;
        }
    }
}
auto OpenGLVolumeBlockCacheImpl::GetMappingTable() -> const std::vector<uint32_t> &
{
    return mapping_table;
}
auto OpenGLVolumeBlockCacheImpl::GetLodMappingTableOffset() -> const std::map<uint32_t, uint32_t> &
{
    return lod_mapping_table_offset;
}
auto OpenGLVolumeBlockCacheImpl::GetOpenGLTextureHandles() -> std::vector<uint32_t>
{
    return this->gl_textures;
}
void OpenGLVolumeBlockCacheImpl::createBlockCacheTable()
{
    for(uint32_t t=0;t<gl_tex_num;t++){
        for(uint32_t k=0;k<gl_tex_shape[2]/block_length;k++){
            for(uint32_t j=0;j<gl_tex_shape[1]/block_length;j++){
                for(uint32_t i=0;i<gl_tex_shape[0]/block_length;i++){
                    BlockCacheItem item;
                    item.pos_index={i,j,k,t};
                    item.block_index={INVALID,INVALID,INVALID,INVALID};
                    item.valid=false;
                    item.cached=false;
                    block_cache_table.push_back(item);
                }
            }
        }
    }
}
void OpenGLVolumeBlockCacheImpl::updateMappingTable(const std::array<uint32_t, 4> &index,
                                                    const std::array<uint32_t, 4> &pos, bool valid)
{
    size_t flat_idx;
    try{
        flat_idx=((size_t)index[2]*lod_block_dim.at(index[3])[0]*lod_block_dim.at(index[3])[1]
                  +index[1]*lod_block_dim.at(index[3])[0]
                  +index[0])*4+lod_mapping_table_offset.at(index[3]);
        mapping_table.at(flat_idx+0)=pos[0];
        mapping_table.at(flat_idx+1)=pos[1];
        mapping_table.at(flat_idx+2)=pos[2];
        if(valid)
            mapping_table.at(flat_idx+3)=pos[3]|(0x00010000);
        else
            mapping_table.at(flat_idx+3)&=0x0000ffff;
    }
    catch (const std::exception& err) {
        std::cout<<index[0]<<" "<<index[1]<<" "<<index[2]<<" "<<index[3]<<"\n"
                 <<pos[0]<<" "<<pos[1]<<" "<<pos[2]<<" "<<pos[3]<<" "<<flat_idx<<std::endl;
        spdlog::error("{0}:{1}",__FUNCTION__ ,err.what());
        spdlog::error("index {0} {1} {2} {3}, pos {4} {5} {6} {7}, flag_idx {8}",
                      index[0],index[1],index[2],index[3],
                      pos[0],pos[1],pos[2],pos[3],flat_idx);
        exit(-1);
    }
}
bool OpenGLVolumeBlockCacheImpl::getCachedPos(const std::array<uint32_t, 4> &target, std::array<uint32_t, 4> &pos)
{
    for(const auto& it:block_cache_table){
        if(it.block_index==target && it.cached){
//            assert(!it.valid);
            spdlog::info("Copy CUDA device memory to CUDA Array which already stored.");
            pos=it.pos_index;
            return true;
        }
    }
    for(const auto& it:block_cache_table){
        if(!it.valid && !it.cached){
            pos=it.pos_index;
            return false;
        }
    }
    for(const auto& it:block_cache_table){
        if(!it.valid){
            pos=it.pos_index;
            return false;
        }
    }
    throw std::runtime_error("Can't find empty pos in OpenGL Textures");
}

VS_END
