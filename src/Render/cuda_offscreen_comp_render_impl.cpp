//
// Created by wyz on 2021/9/13.
//

#include <algorithm>

#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/timer.hpp>
#include <VolumeSlicer/Utils/hash.hpp>
#include <VolumeSlicer/memory_helper.hpp>

#include "Render/cuda_offscreen_comp_render_impl.hpp"
#include "Render/cuda_offscreen_comp_render_impl.cuh"
#include "Render/transfer_function_impl.hpp"

#define OFF_RENDER_PARAMETER_VOXEL 0.5f
#define OFF_RENDER_PARAMETER_STEP_RATIO 0.2f

VS_START

std::unique_ptr<CUDAOffScreenCompVolumeRenderer> vs::CUDAOffScreenCompVolumeRenderer::Create(int w, int h, CUcontext ctx)
{
    return std::make_unique<CUDAOffScreenCompVolumeRendererImpl>(w, h, ctx);
}

CUDAOffScreenCompVolumeRendererImpl::CUDAOffScreenCompVolumeRendererImpl(int w, int h, CUcontext ctx) : window_w(w), window_h(h), step(0.f), steps(0)
{
    if (ctx)
    {
        this->cu_context = ctx;
    }
    else
    {
        this->cu_context = GetCUDACtx();
        if (!cu_context)
        {
            throw std::runtime_error("CUDAOffScreenCompVolumeRendererImpl constructor: cu_context is nullptr");
        }
    }
    CUDAOffScreenCompVolumeRendererImpl::resize(w, h);
    CompRenderPolicy default_policy;
    default_policy.lod_dist[0] = std::numeric_limits<float>::max();
    CUDAOffScreenCompVolumeRendererImpl::SetRenderPolicy(default_policy);
}

void CUDAOffScreenCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume)
{
    this->comp_volume = comp_volume;

    this->volume_block_cache = CUDAVolumeBlockCache::Create(this->cu_context);
    this->volume_block_cache->SetCacheBlockLength(comp_volume->GetBlockLength()[0]);
    this->volume_block_cache->SetCacheCapacity(18, 1024, 1024, 1024);
    this->volume_block_cache->CreateMappingTable(this->comp_volume->GetBlockDim());
    uint32_t max_lod = 0, min_lod = 0x0fffffff;
    {
        auto &mapping_table = this->volume_block_cache->GetMappingTable();
        CUDAOffRenderer::UploadMappingTable(mapping_table.data(), mapping_table.size());
        auto &lod_mapping_table_offset = this->volume_block_cache->GetLodMappingTableOffset();

        for (auto &it : lod_mapping_table_offset)
        {
            if (it.first > max_lod)
                max_lod = it.first;
            if (it.first < min_lod)
                min_lod = it.first;
        }
        max_lod--;
        std::vector<uint32_t> offset; // for one block not for uint32_t
        offset.resize(max_lod + 1, 0);
        for (auto &it : lod_mapping_table_offset)
        {
            if (it.first <= max_lod)
                offset.at(it.first) = it.second / 4;
        }
        CUDAOffRenderer::UploadLodMappingTableOffset(offset.data(), offset.size());
    }

    CUDAOffRenderer::CompVolumeParameter compVolumeParameter;
    auto block_length = comp_volume->GetBlockLength();
    auto block_dim = comp_volume->GetBlockDim(0);
    compVolumeParameter.min_lod = min_lod;
    compVolumeParameter.max_lod = max_lod;
    compVolumeParameter.block_length = block_length[0];
    compVolumeParameter.padding = block_length[1];
    compVolumeParameter.no_padding_block_length = block_length[0] - 2 * block_length[1];
    compVolumeParameter.voxel = OFF_RENDER_PARAMETER_VOXEL;
    compVolumeParameter.block_dim = make_int3(block_dim[0], block_dim[1], block_dim[2]);
    {
        int num;
        MemoryHelper::GetRecommendGPUTextureNum<uint8_t>(num);
        compVolumeParameter.volume_texture_shape = make_int4(MemoryHelper::DefaultGPUTextureSizeX,
                                                             MemoryHelper::DefaultGPUTextureSizeY,
                                                             MemoryHelper::DefaultGPUTextureSizeZ, num);
    }
    compVolumeParameter.volume_dim = make_int3(comp_volume->GetVolumeDimX(), comp_volume->GetVolumeDimY(), comp_volume->GetVolumeDimZ());
    compVolumeParameter.volume_space = make_float3(comp_volume->GetVolumeSpaceX(), comp_volume->GetVolumeSpaceY(), comp_volume->GetVolumeSpaceZ());
    compVolumeParameter.volume_board = make_float3(comp_volume->GetVolumeDimX() * comp_volume->GetVolumeSpaceX(),
                                                   comp_volume->GetVolumeDimY() * comp_volume->GetVolumeSpaceY(),
                                                   comp_volume->GetVolumeDimZ() * comp_volume->GetVolumeSpaceZ());
    CUDAOffRenderer::UploadCompVolumeParameter(compVolumeParameter);
    this->step = (std::min)({comp_volume->GetVolumeSpaceX(), comp_volume->GetVolumeSpaceY(), comp_volume->GetVolumeSpaceZ()}) * OFF_RENDER_PARAMETER_STEP_RATIO;
    auto texes = this->volume_block_cache->GetCUDATextureObjects();
    CUDAOffRenderer::SetCUDATextureObject(texes.data(), texes.size());
}

void CUDAOffScreenCompVolumeRendererImpl::SetRenderPolicy(CompRenderPolicy policy)
{
    CUDAOffRenderer::CUDAOffCompRenderPolicy renderPolicy;
    std::copy(policy.lod_dist, policy.lod_dist + 10, renderPolicy.lod_dist);
    CUDAOffRenderer::UploadCUDAOffCompRenderPolicy(renderPolicy);
}

void CUDAOffScreenCompVolumeRendererImpl::SetMPIRender(MPIRenderParameter mpi_render)
{
    LOG_ERROR("This function is not implement for CUDA-OffScreen-CompVolume-Renderer");
}

void CUDAOffScreenCompVolumeRendererImpl::SetStep(double step, int steps)
{
    this->step = step;
    this->steps = steps;
}

void CUDAOffScreenCompVolumeRendererImpl::SetCamera(Camera camera)
{
    this->camera = camera;
}

void CUDAOffScreenCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf)
{
    TransferFuncImpl tf_impl(tf);
    CUDAOffRenderer::UploadTransferFunc(tf_impl.getTransferFunction().data());
    CUDAOffRenderer::UploadPreIntTransferFunc(tf_impl.getPreIntTransferFunc().data());

    CUDAOffRenderer::ShadingParameter shadingParameter;
    shadingParameter.ka = 0.35f;
    shadingParameter.kd = 0.55f;
    shadingParameter.ks = 0.2f;
    shadingParameter.shininess = 36.f;
    CUDAOffRenderer::UploadShadingParameter(shadingParameter);
}

void CUDAOffScreenCompVolumeRendererImpl::render(bool sync)
{
    CUDAOffRenderer::CUDAOffCompRenderParameter cudaOffCompRenderParameter;
    cudaOffCompRenderParameter.image_w = window_w;
    cudaOffCompRenderParameter.image_h = window_h;
    cudaOffCompRenderParameter.fov = camera.zoom;
    cudaOffCompRenderParameter.step = step;
    cudaOffCompRenderParameter.camera_pos = make_float3(camera.pos[0], camera.pos[1], camera.pos[2]);
    cudaOffCompRenderParameter.front = normalize(make_float3(camera.look_at[0] - camera.pos[0], camera.look_at[1] - camera.pos[1], camera.look_at[2] - camera.pos[2]));
    cudaOffCompRenderParameter.up = make_float3(camera.up[0], camera.up[1], camera.up[2]);
    cudaOffCompRenderParameter.right = make_float3(camera.right[0], camera.right[1], camera.right[2]);
    CUDAOffRenderer::UploadCUDAOffCompRenderParameter(cudaOffCompRenderParameter);
    // 1.generate ray_directions
    // 2.generate ray_start_pos and ray_stop_pos according to ray_directions and box of volume
    CUDAOffRenderer::CUDARenderPrepare(window_w, window_h);

    auto block_length = comp_volume->GetBlockLength();
    auto volume_space = make_float3(comp_volume->GetVolumeSpaceX(), comp_volume->GetVolumeSpaceY(), comp_volume->GetVolumeSpaceZ());
    int3 center_block = make_int3(cudaOffCompRenderParameter.camera_pos / volume_space / (block_length[0] - 2 * block_length[1]));

    std::unordered_map<std::array<uint32_t, 4>, int> m;
    // 3.render pass
    int turn = 0;
    while (++turn)
    {

        std::unordered_set<int4, Hash_Int4> missed_blocks;
        assert(missed_blocks.empty());
        // 3.1 render one pass
        {
            AutoTimer timer;
            CUDAOffRenderer::CUDARender(missed_blocks);
        }
        spdlog::set_level(spdlog::level::err);
        // 3.2 process missed blocks
        if (missed_blocks.empty())
        {
            break;
        }
        auto dummy_missed_blocks = missed_blocks;

        auto UploadMissedBlockData = [this, &missed_blocks, &dummy_missed_blocks, &m]() {
            for (auto &block : dummy_missed_blocks)
            {
                auto volume_block = comp_volume->GetBlock({(uint32_t)block.x, (uint32_t)block.y, (uint32_t)block.z, (uint32_t)block.w});
                if (volume_block.valid)
                {
                    volume_block_cache->UploadVolumeBlock(volume_block.index, volume_block.block_data->GetDataPtr(), volume_block.block_data->GetSize(), true);
                    m[volume_block.index] += 1;
                    volume_block.Release();
                    missed_blocks.erase(block);
                }
            }
        };
        if (missed_blocks.size() > volume_block_cache->GetRemainEmptyBlock())
        {
            volume_block_cache->clear();
            int i = 0, n = volume_block_cache->GetRemainEmptyBlock();
            std::vector<int4> sorted_missed_blocks(missed_blocks.size());
            std::copy(missed_blocks.begin(), missed_blocks.end(), sorted_missed_blocks.begin());
            std::sort(sorted_missed_blocks.begin(), sorted_missed_blocks.end(), [center_block](const int4 &v1, const int4 &v2) {
                if (v1.w == v2.w)
                {
                    int d1 = (v1.x - center_block.x) * (v1.x - center_block.x) + (v1.y - center_block.y) * (v1.y - center_block.y) + (v1.z - center_block.z) * (v1.z - center_block.z);
                    int d2 = (v2.x - center_block.x) * (v2.x - center_block.x) + (v2.y - center_block.y) * (v2.y - center_block.y) + (v2.z - center_block.z) * (v2.z - center_block.z);
                    return d1 < d2;
                }
                else
                {
                    return v1.w < v2.w;
                }
            });
            for (auto &block : sorted_missed_blocks)
            {
                this->comp_volume->SetRequestBlock({(uint32_t)block.x, (uint32_t)block.y, (uint32_t)block.z, (uint32_t)block.w});
                if (++i >= n)
                {
                    break;
                }
            }
            while (!missed_blocks.empty() && volume_block_cache->GetRemainEmptyBlock() > 0)
            {
                UploadMissedBlockData();
            }
        }
        else
        {
            for (auto &block : dummy_missed_blocks)
            {
                this->comp_volume->SetRequestBlock({(uint32_t)block.x, (uint32_t)block.y, (uint32_t)block.z, (uint32_t)block.w});
            }
            while (!missed_blocks.empty())
            {
                UploadMissedBlockData();
            }
        }
        {
            auto &m = this->volume_block_cache->GetMappingTable();
            CUDAOffRenderer::UploadMappingTable(m.data(), m.size());
        }
        spdlog::set_level(spdlog::level::info);
        LOG_INFO("current render turn {0} load block num: {3}, total missed block num: {1}, remain missed block num: {2}, ",
                 turn, dummy_missed_blocks.size(), missed_blocks.size(),dummy_missed_blocks.size() - missed_blocks.size());
    }
    CUDAOffRenderer::GetRenderImage(reinterpret_cast<uint8_t *>(image.GetData()));
    spdlog::set_level(spdlog::level::info);
    LOG_INFO("CUDA comp-volume render finish.");
    LOG_INFO("Total upload block set's size is: {0}.", m.size());
    LOG_INFO("Print block upload info:");
    int cnt = 0;
    for (auto &it : m)
    {
        LOG_INFO("block ({0},{1},{2},{3}) upload count {4}.", it.first[0], it.first[1], it.first[2], it.first[3], it.second);
        cnt += it.second;
    }
    LOG_INFO("Total upload block num is: {0}.", cnt);
    LOG_INFO("Multi-upload block num is: {0}.", cnt - m.size());
}
auto CUDAOffScreenCompVolumeRendererImpl::GetImage() -> const Image<Color4b> &
{
    return image;
}

void CUDAOffScreenCompVolumeRendererImpl::resize(int w, int h)
{
    this->window_w = w;
    this->window_h = h;

    image = Image<Color4b>(w, h);
}
void CUDAOffScreenCompVolumeRendererImpl::clear()
{
    volume_block_cache.reset();
    comp_volume.reset();
}
auto CUDAOffScreenCompVolumeRendererImpl::GetBackendName() -> std::string
{
    return "cuda";
}

VS_END
