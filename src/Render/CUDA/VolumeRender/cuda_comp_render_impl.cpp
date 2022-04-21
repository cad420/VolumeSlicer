//
// Created by wyz on 2021/7/21.
//

#include <chrono>
#include <iostream>

#include <VolumeSlicer/Algorithm/memory_helper.hpp>

#include "Common/helper_math.h"
#include "Common/transfer_function_impl.hpp"
#include "cuda_comp_render_impl.hpp"
#include "cuda_comp_render_impl.cuh"

#define START_CPU_TIMER                                                                                                \
    {                                                                                                                  \
        auto _start = std::chrono::steady_clock::now();

#define END_CPU_TIMER                                                                                                  \
    auto _end = std::chrono::steady_clock::now();                                                                      \
    auto _t = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start);                                    \
    std::cout << "CPU cost time : " << _t.count() << "ms" << std::endl;                                                \
    }

#define START_CUDA_DRIVER_TIMER                                                                                        \
    CUevent start, stop;                                                                                               \
    float elapsed_time;                                                                                                \
    cuEventCreate(&start, CU_EVENT_DEFAULT);                                                                           \
    cuEventCreate(&stop, CU_EVENT_DEFAULT);                                                                            \
    cuEventRecord(start, 0);

#define STOP_CUDA_DRIVER_TIMER                                                                                         \
    cuEventRecord(stop, 0);                                                                                            \
    cuEventSynchronize(stop);                                                                                          \
    cuEventElapsedTime(&elapsed_time, start, stop);                                                                    \
    cuEventDestroy(start);                                                                                             \
    cuEventDestroy(stop);                                                                                              \
    std::cout << "GPU cost time: " << elapsed_time << "ms" << std::endl;

#define START_CUDA_RUNTIME_TIMER                                                                                       \
    {                                                                                                                  \
        cudaEvent_t start, stop;                                                                                       \
        float elapsedTime;                                                                                             \
        (cudaEventCreate(&start));                                                                                     \
        (cudaEventCreate(&stop));                                                                                      \
        (cudaEventRecord(start, 0));

#define STOP_CUDA_RUNTIME_TIMER                                                                                        \
    (cudaEventRecord(stop, 0));                                                                                        \
    (cudaEventSynchronize(stop));                                                                                      \
    (cudaEventElapsedTime(&elapsedTime, start, stop));                                                                 \
    printf("\tGPU cost time used: %.f ms\n", elapsedTime);                                                             \
    (cudaEventDestroy(start));                                                                                         \
    (cudaEventDestroy(stop));                                                                                          \
    }

VS_START

std::unique_ptr<CUDACompVolumeRenderer> CUDACompVolumeRenderer::Create(int w, int h, CUcontext ctx)
{
    return std::make_unique<CUDACompVolumeRendererImpl>(w, h, ctx);
}

CUDACompVolumeRendererImpl::CUDACompVolumeRendererImpl(int w, int h, CUcontext ctx)
    : window_w(w), window_h(h), steps(0), step(0.f), mpi_render(false)
{
    if (ctx)
    {
        this->cu_context = ctx;
    }
    else
    {
        this->cu_context = GetCUDACtx();
        if (!cu_context)
            throw std::runtime_error("cu_context is nullptr");
    }
    CUDACompVolumeRendererImpl::resize(w, h);
}
CUDARenderer::CompVolumeParameter g_compVolumeParameter;
void CUDACompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume)
{
    this->comp_volume = comp_volume;

    this->cuda_volume_block_cache = CUDAVolumeBlockCache::Create();
    this->cuda_volume_block_cache->SetCacheBlockLength(comp_volume->GetBlockLength()[0]);
    {
        int num;
        MemoryHelper::GetRecommendGPUTextureNum<uint8_t>(num);
        LOG_INFO("CUDACompVolumeRenderer: Recommend GPU texture num({})",num);
        this->cuda_volume_block_cache->SetCacheCapacity(num, MemoryHelper::DefaultGPUTextureSizeX, MemoryHelper::DefaultGPUTextureSizeY, MemoryHelper::DefaultGPUTextureSizeZ);
    }
    this->cuda_volume_block_cache->CreateMappingTable(this->comp_volume->GetBlockDim());
    //  /4 represent for block not for uint32_t
    this->missed_blocks_pool.resize(this->cuda_volume_block_cache->GetMappingTable().size() / 4, 0);
    uint32_t max_lod = 0, min_lod = 0xffffffff;
    {
        auto &mapping_table = this->cuda_volume_block_cache->GetMappingTable();
        CUDARenderer::UploadMappingTable(mapping_table.data(), mapping_table.size());
        auto &lod_mapping_table_offset = this->cuda_volume_block_cache->GetLodMappingTableOffset();

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
        CUDARenderer::UploadLodMappingTableOffset(offset.data(), offset.size());
        block_offset = std::move(offset);
    }

    CUDARenderer::CompVolumeParameter compVolumeParameter;
    auto block_length = comp_volume->GetBlockLength();
    compVolumeParameter.min_lod = min_lod;
    compVolumeParameter.max_lod = max_lod;
    compVolumeParameter.block_length = block_length[0];
    compVolumeParameter.padding = block_length[1];
    compVolumeParameter.no_padding_block_length = block_length[0] - 2 * block_length[1];
    auto block_dim = comp_volume->GetBlockDim(0);
    compVolumeParameter.block_dim = make_int3(block_dim[0], block_dim[1], block_dim[2]);
    {
        auto shape = cuda_volume_block_cache->GetCacheShape();
        compVolumeParameter.texture_shape = make_int4(shape[1], shape[2], shape[3], shape[0]);
    }
    compVolumeParameter.volume_board = make_float3(comp_volume->GetVolumeDimX() * comp_volume->GetVolumeSpaceX(),
                                                   comp_volume->GetVolumeDimY() * comp_volume->GetVolumeSpaceY(),
                                                   comp_volume->GetVolumeDimZ() * comp_volume->GetVolumeSpaceZ());
    compVolumeParameter.volume_dim = make_int3(comp_volume->GetVolumeDimX(),
                                               comp_volume->GetVolumeDimY(),
                                               comp_volume->GetVolumeDimZ());
    CUDARenderer::UploadCompVolumeParameter(compVolumeParameter);
    g_compVolumeParameter = compVolumeParameter;

    auto texes = this->cuda_volume_block_cache->GetCUDATextureObjects();
    CUDARenderer::SetCUDATextureObject(texes.data(), texes.size());
}

void CUDACompVolumeRendererImpl::SetRenderPolicy(CompRenderPolicy policy)
{
    {
        CUDARenderer::CUDACompRenderPolicy renderPolicy;
        std::copy(policy.lod_dist,policy.lod_dist+10,renderPolicy.lod_dist);
        CUDARenderer::UploadCUDACompRenderPolicy(renderPolicy);
    }
    if (!policy.cdf_value_file.empty())
    {
        try
        {
            cdf_manager = std::make_unique<CDFManager>(policy.cdf_value_file.c_str());
        }
        catch (std::exception const &err)
        {
            LOG_ERROR(err.what());
            return;
        }
        cdf_block_length = cdf_manager->GetCDFBlockLength();
        cdf_dim_x = cdf_manager->GetBlockCDFDim()[0];
        cdf_dim_y = cdf_manager->GetBlockCDFDim()[1];
        cdf_dim_z = cdf_manager->GetBlockCDFDim()[2];
        auto &cdf_map = cdf_manager->GetCDFMap();

        LOG_INFO("cdf_block_length: {0}, cdf_dim: {1} {2} {3}", cdf_block_length, cdf_dim_x, cdf_dim_y, cdf_dim_z);
        g_compVolumeParameter.cdf_block_length = cdf_block_length;
        assert(cdf_dim_x == cdf_dim_y && cdf_dim_y == cdf_dim_z);
        g_compVolumeParameter.cdf_dim_len = cdf_dim_x;
        g_compVolumeParameter.cdf_block_num = cdf_dim_x * cdf_dim_y * cdf_dim_z;
        CUDARenderer::UploadCompVolumeParameter(g_compVolumeParameter);

        std::vector<std::pair<std::array<uint32_t, 4>, std::vector<uint32_t>>> array;
        array.reserve(cdf_map.size());
        for (auto &it : cdf_map)
        {
            array.push_back(it);
        }
        cdf_manager.reset();

        std::sort(array.begin(), array.end(), [](const auto &idx1, const auto &idx2) {
            if (idx1.first[3] == idx2.first[3])
            {
                if (idx1.first[2] == idx2.first[2])
                {
                    if (idx1.first[1] == idx2.first[1])
                    {
                        return idx1.first[0] < idx2.first[0];
                    }
                    else
                        return idx1.first[1] < idx2.first[1];
                }
                else
                    return idx1.first[2] < idx2.first[2];
            }
            else
                return idx1.first[3] < idx2.first[3];
        });

        std::unordered_map<uint32_t, std::vector<uint32_t>> tmp;

        for (auto &it : array)
        {
            auto &v = tmp[it.first[3]];
            v.insert(v.end(), it.second.begin(), it.second.end());
        }

        std::vector<const uint32_t *> data(tmp.size());
        std::vector<size_t> size(tmp.size());
        for (int i = 0; i < data.size(); i++)
        {
            data[i] = tmp[i].data();
            size[i] = tmp[i].size();
            LOG_INFO("lod {0} has cdf size: {1}", i, size[i]);
        }

        CUDARenderer::UploadCDFMap(data.data(), data.size(), size.data());
    }
}

void CUDACompVolumeRendererImpl::SetMPIRender(MPIRenderParameter mpiRenderParameter)
{
    CUDARenderer::UploadMPIRenderParameter(mpiRenderParameter);
    this->mpi_render = true;
}

void CUDACompVolumeRendererImpl::SetStep(double step, int steps)
{
    this->step = step;
    this->steps = steps;
}

void CUDACompVolumeRendererImpl::SetCamera(Camera camera)
{
    this->camera = camera;
}

void CUDACompVolumeRendererImpl::SetTransferFunc(TransferFunc tf)
{
    TransferFuncImpl tf_impl(tf);
    CUDARenderer::UploadTransferFunc(tf_impl.getTransferFunction().data());
    CUDARenderer::UploadPreIntTransferFunc(tf_impl.getPreIntTransferFunc().data());

    // todo move to another place
    CUDARenderer::LightParameter lightParameter;
    lightParameter.bg_color = make_float4(0.f, 0.f, 0.f, 0.f);
    lightParameter.ka = 0.35f;
    lightParameter.kd = 0.75f;
    lightParameter.ks = 0.3f;
    lightParameter.shininess = 36.f;
    CUDARenderer::UploadLightParameter(lightParameter);
}

void CUDACompVolumeRendererImpl::render(bool sync)
{

    // may change every time render
    CUDARenderer::CUDACompRenderParameter cudaCompRenderParameter;
    cudaCompRenderParameter.w = window_w;
    cudaCompRenderParameter.h = window_h;
    cudaCompRenderParameter.fov = camera.zoom;
    cudaCompRenderParameter.step = this->step;
    cudaCompRenderParameter.steps = this->steps;
    cudaCompRenderParameter.view_pos = make_float3(camera.pos[0], camera.pos[1], camera.pos[2]);
    cudaCompRenderParameter.view_direction = normalize(make_float3(
        camera.look_at[0] - camera.pos[0], camera.look_at[1] - camera.pos[1], camera.look_at[2] - camera.pos[2]));
    cudaCompRenderParameter.up = make_float3(camera.up[0], camera.up[1], camera.up[2]);
    cudaCompRenderParameter.right = make_float3(camera.right[0], camera.right[1], camera.right[2]);
    cudaCompRenderParameter.space =
        make_float3(comp_volume->GetVolumeSpaceX(), comp_volume->GetVolumeSpaceY(), comp_volume->GetVolumeSpaceZ());
    cudaCompRenderParameter.mpi_render = this->mpi_render;
    CUDARenderer::UploadCUDACompRenderParameter(cudaCompRenderParameter);

    //    START_CPU_TIMER
    calcMissedBlocks();

    filterMissedBlocks();

    sendRequests();

    do
    {
        fetchBlocks(sync);
    } while (sync && !isRenderFinish());

    if (sync)
    {
        assert(isRenderFinish());
    }

    clearCurrentInfo();
    //    END_CPU_TIMER

    auto &m = this->cuda_volume_block_cache->GetMappingTable();
    CUDARenderer::UploadMappingTable(m.data(), m.size());

    START_CUDA_RUNTIME_TIMER
    CUDARenderer::CUDARender(window_w, window_h, reinterpret_cast<uint8_t *>(image.GetData()));
    STOP_CUDA_RUNTIME_TIMER
}

void CUDACompVolumeRendererImpl::calcMissedBlocks()
{

    std::unordered_set<std::array<uint32_t, 4>> cur_missed_blocks;
    START_CUDA_RUNTIME_TIMER
    CUDARenderer::CUDACalcBlock(missed_blocks_pool.data(), missed_blocks_pool.size(), window_w, window_h);
    STOP_CUDA_RUNTIME_TIMER
    for (uint32_t lod = 0; lod < block_offset.size(); lod++)
    {
        auto lod_block_dim = comp_volume->GetBlockDim(lod);
        int cnt = 0;
        for (size_t idx = block_offset[lod];
             idx < (lod + 1 < block_offset.size() ? block_offset[lod + 1] : missed_blocks_pool.size()); idx++)
        {
            cnt++;
            if (!missed_blocks_pool[idx])
                continue;

            size_t index = idx - block_offset[lod];
            uint32_t z = index / lod_block_dim[0] / lod_block_dim[1];
            uint32_t y = (index - z * lod_block_dim[0] * lod_block_dim[1]) / lod_block_dim[0];
            uint32_t x = (index % (lod_block_dim[0] * lod_block_dim[1])) % lod_block_dim[0];
            cur_missed_blocks.insert({x, y, z, lod});
        }

    }

    for (auto &it : cur_missed_blocks)
    {
        if (missed_blocks.find(it) == missed_blocks.end())
        {
            new_missed_blocks.insert(it);
        }
    }

    for (auto &it : missed_blocks)
    {
        if (cur_missed_blocks.find(it) == cur_missed_blocks.end())
        {
            no_missed_blocks.insert(it);
        }
    }

    this->missed_blocks = std::move(cur_missed_blocks);
}

void CUDACompVolumeRendererImpl::filterMissedBlocks()
{
    if (!new_missed_blocks.empty())
    {
        std::unordered_set<std::array<uint32_t, 4>> tmp;
        for (auto &it : new_missed_blocks)
        {
            bool cached = this->cuda_volume_block_cache->SetCachedBlockValid(it);
            if (cached)
            {
            }
            else
            {
                tmp.insert(it);
            }
        }
        new_missed_blocks = std::move(tmp);
    }

    if (!no_missed_blocks.empty())
    {
        for (auto &it : no_missed_blocks)
        {
            this->cuda_volume_block_cache->SetBlockInvalid(it);
        }
    }
}

void CUDACompVolumeRendererImpl::sendRequests()
{
    this->comp_volume->PauseLoadBlock();
    {
        if (!missed_blocks.empty())
        {
            std::vector<std::array<uint32_t, 4>> targets;
            targets.reserve(missed_blocks.size());
            for (auto &it : missed_blocks)
                targets.push_back(it);
            comp_volume->ClearBlockInQueue(targets);
        }
        for (auto &it : new_missed_blocks)
        {
            comp_volume->SetRequestBlock(it);
        }

        for (auto &it : no_missed_blocks)
        {
            comp_volume->EraseBlockInRequest(it);
        }
    }
    this->comp_volume->StartLoadBlock();
}

void CUDACompVolumeRendererImpl::clearCurrentInfo()
{
    new_missed_blocks.clear();
    no_missed_blocks.clear();
}

void CUDACompVolumeRendererImpl::fetchBlocks(bool sync)
{
    this->is_render_finish = true;
    if (sync)
    {
        auto tmp = new_missed_blocks;
        for (auto &it : tmp)
        {
            auto block = comp_volume->GetBlock(it);
            if (block.valid)
            {
                assert(block.block_data->GetDataPtr());
                this->cuda_volume_block_cache->UploadVolumeBlock(block.index, block.block_data->GetDataPtr(),
                                                                 block.block_data->GetSize(), true);
                block.Release();
                new_missed_blocks.erase(it);
            }
            else
            {
                this->is_render_finish = false;
            }
        }
    }
    else
    {
        for (auto &it : missed_blocks)
        {
            auto block = comp_volume->GetBlock(it);
            if (block.valid)
            {
                assert(block.block_data->GetDataPtr());
                this->cuda_volume_block_cache->UploadVolumeBlock(block.index, block.block_data->GetDataPtr(),
                                                                 block.block_data->GetSize(), true);
                block.Release();
            }
            else
            {
                this->is_render_finish = false;
            }
        }
    }
}

auto CUDACompVolumeRendererImpl::GetImage() -> const Image<Color4b> &
{
    return image;
}

void CUDACompVolumeRendererImpl::resize(int w, int h)
{
    if (w < 0 || h < 0 || w > 10000 || h > 10000)
    {
        LOG_ERROR("error w({0}) or h({1}) for cuda comp volume renderer.", w, h);
        return;
    }
    image = Image<Color4b>(w, h);
}

void CUDACompVolumeRendererImpl::clear()
{
}

auto CUDACompVolumeRendererImpl::GetBackendName() -> std::string
{
    return "cuda";
}

bool CUDACompVolumeRendererImpl::isRenderFinish()
{
    return is_render_finish;
}

CUDACompVolumeRendererImpl::~CUDACompVolumeRendererImpl()
{
    CUDARenderer::DeleteAllCUDAResources();
    cuda_volume_block_cache.reset(nullptr);
}

VS_END
