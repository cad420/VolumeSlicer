//
// Created by wyz on 2021/9/13.
//
#include "cuda_offscreen_comp_render_impl.hpp"
#include <Utils/logger.hpp>
#include "transfer_function_impl.hpp"
#include "cuda_offscreen_comp_render_impl.cuh"


VS_START

std::unique_ptr<CUDAOffScreenCompVolumeRenderer> vs::CUDAOffScreenCompVolumeRenderer::Create(int w, int h,CUcontext ctx)
{
    return std::make_unique<CUDAOffScreenCompVolumeRendererImpl>(w,h,ctx);
}

CUDAOffScreenCompVolumeRendererImpl::CUDAOffScreenCompVolumeRendererImpl(int w, int h,CUcontext ctx)
:window_w(w),window_h(h),step(0.f),steps(0)
{
    if(ctx){
       this->cu_context=ctx;
    }
    else{
        this->cu_context=GetCUDACtx();
        if(!cu_context){
            throw std::runtime_error("CUDAOffScreenCompVolumeRendererImpl constructor: cu_context is nullptr");
        }
    }
    CUDAOffScreenCompVolumeRendererImpl::resize(w,h);
}

void CUDAOffScreenCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume)
{
    this->comp_volume        = comp_volume;

    this->volume_block_cache = CUDAVolumeBlockCache::Create(this->cu_context);
    this->volume_block_cache->SetCacheBlockLength(comp_volume->GetBlockLength()[0]);
    this->volume_block_cache->SetCacheCapacity(12,1024,1024,1024);
    this->volume_block_cache->CreateMappingTable(this->comp_volume->GetBlockDim());
    uint32_t max_lod = 0,min_lod=0x0fffffff;
    {
        auto &mapping_table = this->volume_block_cache->GetMappingTable();
        CUDAOffRenderer::UploadMappingTable(mapping_table.data(), mapping_table.size());
        auto &lod_mapping_table_offset = this->volume_block_cache->GetLodMappingTableOffset();

        for (auto &it :lod_mapping_table_offset) {
            if (it.first > max_lod) max_lod = it.first;
            if (it.first < min_lod) min_lod = it.first;
        }
        max_lod--;
        std::vector<uint32_t> offset;//for one block not for uint32_t
        offset.resize(max_lod+1, 0);
        for (auto &it :lod_mapping_table_offset) {
            if(it.first<=max_lod)
                offset.at(it.first) = it.second/4;
        }
        CUDAOffRenderer::UploadLodMappingTableOffset(offset.data(), offset.size());
    }

    CUDAOffRenderer::CompVolumeParameter compVolumeParameter;
    auto block_length = comp_volume->GetBlockLength();
    auto block_dim    = comp_volume->GetBlockDim(0);
    compVolumeParameter.min_lod                 = min_lod;
    compVolumeParameter.max_lod                 = max_lod;
    compVolumeParameter.block_length            = block_length[0];
    compVolumeParameter.padding                 = block_length[1];
    compVolumeParameter.no_padding_block_length = block_length[0]-2*block_length[1];
    compVolumeParameter.voxel                   = 1.f;
    compVolumeParameter.block_dim               = make_int3(block_dim[0],block_dim[1],block_dim[2]);
    compVolumeParameter.volume_texture_shape    = make_int4(1024,1024,1024,12);
    compVolumeParameter.volume_dim              = make_int3(comp_volume->GetVolumeDimX(),
                                                            comp_volume->GetVolumeDimY(),
                                                            comp_volume->GetVolumeDimZ());
    compVolumeParameter.volume_space            = make_float3(comp_volume->GetVolumeSpaceX(),
                                                              comp_volume->GetVolumeSpaceY(),
                                                              comp_volume->GetVolumeSpaceZ());
    compVolumeParameter.volume_board            = make_float3(comp_volume->GetVolumeDimX()*comp_volume->GetVolumeSpaceX() ,
                                                              comp_volume->GetVolumeDimY()*comp_volume->GetVolumeSpaceY() ,
                                                              comp_volume->GetVolumeDimZ()*comp_volume->GetVolumeSpaceZ());
    CUDAOffRenderer::UploadCompVolumeParameter(compVolumeParameter);
    this->step                                  = (std::min)({comp_volume->GetVolumeSpaceX(),
                                                                    comp_volume->GetVolumeSpaceY(),
                                                                    comp_volume->GetVolumeSpaceZ()})
                                                  * 0.3f;
    auto texes=this->volume_block_cache->GetCUDATextureObjects();
    CUDAOffRenderer::SetCUDATextureObject(texes.data(),texes.size());
}
void CUDAOffScreenCompVolumeRendererImpl::SetMPIRender(MPIRenderParameter)
{
    LOG_ERROR("This function is not implement for CUDA-OffScreen-CompVolume-Renderer");
}
void CUDAOffScreenCompVolumeRendererImpl::SetStep(double step, int steps)
{
    this->step=step;
    this->steps=steps;
}
void CUDAOffScreenCompVolumeRendererImpl::SetCamera(Camera camera)
{
    this->camera=camera;
}
void CUDAOffScreenCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf)
{
    TransferFuncImpl tf_impl(tf);
    CUDAOffRenderer::UploadTransferFunc(tf_impl.getTransferFunction().data());
    CUDAOffRenderer::UploadPreIntTransferFunc(tf_impl.getPreIntTransferFunc().data());

    CUDAOffRenderer::ShadingParameter shadingParameter;
    shadingParameter.ka=0.25f;
    shadingParameter.kd=0.5f;
    shadingParameter.ks=0.36f;
    shadingParameter.shininess=100.f;
    CUDAOffRenderer::UploadShadingParameter(shadingParameter);
}
void CUDAOffScreenCompVolumeRendererImpl::render()
{
    CUDAOffRenderer::CUDAOffCompRenderParameter cudaOffCompRenderParameter;
    cudaOffCompRenderParameter.image_w = window_w;
    cudaOffCompRenderParameter.image_h = window_h;
    cudaOffCompRenderParameter.fov     = camera.zoom;
    cudaOffCompRenderParameter.step    = step;
    cudaOffCompRenderParameter.camera_pos = make_float3(camera.pos[0],camera.pos[1],camera.pos[2]);
    cudaOffCompRenderParameter.front      = normalize(make_float3(camera.look_at[0]-camera.pos[0],
                                                                     camera.look_at[1]-camera.pos[1],
                                                                     camera.look_at[2]-camera.pos[2]));
    cudaOffCompRenderParameter.up         = make_float3(camera.up[0],camera.up[1],camera.up[2]);
    cudaOffCompRenderParameter.right      = make_float3(camera.right[0],camera.right[1],camera.right[2]);
    CUDAOffRenderer::UploadCUDAOffCompRenderParameter(cudaOffCompRenderParameter);
    //1.generate ray_directions
    //2.generate ray_start_pos and ray_stop_pos according to ray_directions and box of volume
    CUDAOffRenderer::CUDARenderPrepare(window_w,window_h);


    //3.render pass
    int turn = 0;
    while(++turn){
        std::unordered_set<int4,Hash_Int4> missed_blocks;
        assert(missed_blocks.empty());
        //3.1 render one pass
        CUDAOffRenderer::CUDARender(missed_blocks);

        //3.2 process missed blocks
        if(missed_blocks.empty()){
            break;
        }
        auto dummy_missed_blocks=missed_blocks;
        for(auto& block:dummy_missed_blocks){
            this->comp_volume->SetRequestBlock({(uint32_t)block.x,(uint32_t)block.y,(uint32_t)block.z,(uint32_t)block.w});
        }
        auto UploadMissedBlockData=[this,&missed_blocks,&dummy_missed_blocks](){
            for(auto& block:dummy_missed_blocks){
                auto volume_block=comp_volume->GetBlock({(uint32_t)block.x,
                                                         (uint32_t)block.y,
                                                         (uint32_t)block.z,
                                                         (uint32_t)block.w});
                if(volume_block.valid){
                    volume_block_cache->UploadVolumeBlock(volume_block.index,
                                                          volume_block.block_data->GetDataPtr(),
                                                          volume_block.block_data->GetSize());
                    volume_block.Release();
                    missed_blocks.erase(block);
                }
            }
        };
        if(missed_blocks.size() > volume_block_cache->GetRemainEmptyBlock()){
            volume_block_cache->clear();
            while(!missed_blocks.empty() && volume_block_cache->GetRemainEmptyBlock()>0){
                UploadMissedBlockData();
            }
        }
        else{
            while(!missed_blocks.empty()){
                UploadMissedBlockData();
            }
        }
        {
            auto& m=this->volume_block_cache->GetMappingTable();
            CUDAOffRenderer::UploadMappingTable(m.data(),m.size());
        }
        LOG_INFO("current render turn {0} load block num: {3}, total missed block num: {1}, remain missed block num: {2}, " ,
                 turn,dummy_missed_blocks.size(),missed_blocks.size(),dummy_missed_blocks.size()-missed_blocks.size());
    }
    CUDAOffRenderer::GetRenderImage(reinterpret_cast<uint8_t*>(image.GetData()));
    LOG_INFO("CUDA comp-volume render finish.");
}
auto CUDAOffScreenCompVolumeRendererImpl::GetImage() -> const Image<Color4b> &
{
    return image;
}
auto CUDAOffScreenCompVolumeRendererImpl::GetFrame() -> const Image<uint32_t> &
{
    return frame;
}
void CUDAOffScreenCompVolumeRendererImpl::resize(int w, int h)
{
    this->window_w=w;
    this->window_h=h;
    frame.width=w;
    frame.height=h;
    frame.data.resize((size_t)w*h);

    image=Image<Color4b>(w,h);
}
void CUDAOffScreenCompVolumeRendererImpl::clear()
{

}

VS_END
