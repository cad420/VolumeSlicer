//
// Created by wyz on 2021/7/21.
//

#include <iostream>

#include <VolumeSlicer/Data/volume_cache.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/timer.hpp>

#include "Common/cuda_utils.hpp"
#include "cuda_volume_cache_impl.hpp"

VS_START

std::unique_ptr<CUDAVolumeBlockCache> CUDAVolumeBlockCache::Create(CUcontext ctx)
{
    return std::make_unique<CUDAVolumeBlockCacheImpl>(ctx);
}
CUDAVolumeBlockCacheImpl::CUDAVolumeBlockCacheImpl(CUcontext ctx)
{
    if (ctx == nullptr)
    {
        this->cu_context = GetCUDACtx();
    }
    else
    {
        this->cu_context = ctx;
    }
    LOG_INFO("Create CUDAVolumeBlockCache.");
}

void CUDAVolumeBlockCacheImpl::SetCacheBlockLength(uint32_t block_length)
{
    this->block_length = block_length;

    this->chunk_cache = std::make_unique<ChunkCache>(block_length*block_length*block_length);
    this->chunk_cache->SetCacheStorage(12);
}

void CUDAVolumeBlockCacheImpl::SetCacheCapacity(uint32_t num, uint32_t x, uint32_t y, uint32_t z)
{
    this->cu_array_num = num;
    this->cu_array_shape = {x, y, z};
    this->cu_arrays.resize(num, nullptr);
    this->cu_textures.resize(num, 0);

    for (uint32_t i = 0; i < cu_array_num; i++)
    {
        CreateCUDATexture3D(make_cudaExtent(x, y, z), &cu_arrays[i], &cu_textures[i]);
    }
    this->createBlockCacheTable();
    LOG_INFO("SetCacheCapacity, num:{0} x:{1} y:{2} z:{3}.", num, x, y, z);
}

void CUDAVolumeBlockCacheImpl::CreateMappingTable(const std::map<uint32_t, std::array<uint32_t, 3>> &lod_block_dim)
{
    this->lod_block_dim = lod_block_dim;
    this->lod_mapping_table_offset[lod_block_dim.begin()->first] = 0;
    this->min_lod = 0xffffffff;
    this->max_lod = 0;
    for (auto it = lod_block_dim.begin(); it != lod_block_dim.end(); it++)
    {
        this->min_lod = it->first < min_lod ? it->first : min_lod;
        this->max_lod = it->first > max_lod ? it->first : max_lod;
        auto &t = it->second;
        size_t lod_block_num = (size_t)t[0] * t[1] * t[2];
        lod_mapping_table_offset[it->first + 1] = lod_mapping_table_offset[it->first] + lod_block_num * 4;
    }
    mapping_table.assign(lod_mapping_table_offset.at(max_lod + 1), 0);
}

static size_t count = 0;
void CUDAVolumeBlockCacheImpl::UploadVolumeBlock(const std::array<uint32_t, 4> &index, uint8_t *data, size_t size,
                                                 bool device)
{
    // upload data to texture
    std::array<uint32_t, 4> pos{INVALID, INVALID, INVALID, INVALID};
    bool cached = getCachedPos(index, pos);

    if (!cached)
    {
        //copy origin data in cuda array to cache
        if(chunk_cache){
            for(auto& it:block_cache_table){
                if(it.pos_index == pos && it.block_index!=index && it.cached && it.block_index[0]!=INVALID){
                    //cuda memory copy from cuda array to cpu memory
                    {
                        auto cacheID = BlockIndexToCacheID(it.block_index);
                        if(chunk_cache->Query(cacheID)){
                            chunk_cache->GetCache(cacheID);//access the cache to change the cache priority
                            break;
                        }
                        SCOPE_TIMER("copy block from cuda array to chunk cache cost time ")
                        auto cache = chunk_cache->GetCacheRef(cacheID);

                        CUDA_MEMCPY3D m{};
                        m.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                        m.srcArray = (CUarray)cu_arrays[pos[3]];
                        m.srcXInBytes = it.pos_index[0]*block_length;
                        m.srcY = it.pos_index[1]*block_length;
                        m.srcZ = it.pos_index[2]*block_length;

                        m.dstMemoryType = CU_MEMORYTYPE_HOST;
                        m.dstHost = cache.data;

                        m.WidthInBytes = block_length;
                        m.Height = block_length;
                        m.Depth = block_length;

                        CUDA_DRIVER_API_CALL(cuMemcpy3D(&m));

                        LOG_DEBUG("Copy block({} {} {} {}) data from cuda texture to chunk cache",
                                 it.block_index[0],it.block_index[1],it.block_index[2],it.block_index[3]);

                    }
                    break;
                }
            }
        }
        if (device)
            UpdateCUDATexture3D(data, cu_arrays[pos[3]], block_length, block_length * pos[0], block_length * pos[1],
                                block_length * pos[2]);
        else
            UpdateCUDATexture3D(data, cu_arrays[pos[3]], block_length, block_length, block_length,
                                block_length * pos[0], block_length * pos[1], block_length * pos[2]);
        LOG_DEBUG("Upload block({0},{1},{2},{3}) to CUDA Array({4},{5},{6},{7})", index[0], index[1], index[2],
                     index[3], pos[0], pos[1], pos[2], pos[3]);
    }
    else
    {
        LOG_DEBUG("UploadVolumeBlock which has already been cached.");
    }
    // update block_cache_table
    for (auto &it : block_cache_table)
    {
        if (it.pos_index == pos)
        {
            if (cached)
            {
                assert(it.cached && it.block_index == index);
            }
            if (it.block_index != index && it.block_index[0] != INVALID)
            {
                this->updateMappingTable(it.block_index, {0, 0, 0, 0}, false);
            }
            it.block_index = index;
            it.valid = true;
            it.cached = true;
            it.t = ++count;
            break;
        }
    }

    // update mapping_table
    updateMappingTable(index, pos);
}

bool CUDAVolumeBlockCacheImpl::IsCachedBlock(const std::array<uint32_t, 4> &target)
{
    for (auto &it : block_cache_table)
    {
        if (it.block_index == target)
        {
            if(it.cached){
                return true;
            }
        }
    }
    //query chunk cache
    if(chunk_cache){
        bool cached = chunk_cache->Query(BlockIndexToCacheID(target));
        if(cached){
            return true;
        }
    }
    return false;

}

bool CUDAVolumeBlockCacheImpl::IsValidBlock(const std::array<uint32_t, 4> &target)
{
    for (auto &it : block_cache_table)
    {
        if (it.block_index == target)
        {
            return it.valid;
        }
    }
}

auto CUDAVolumeBlockCacheImpl::GetBlockStatus(const std::array<uint32_t, 4> &target) -> std::array<bool, 2>
{
    for (auto &it : block_cache_table)
    {
        if (it.block_index == target)
        {
            return {it.valid, it.cached};
        }
    }
}

int CUDAVolumeBlockCacheImpl::GetRemainEmptyBlock() const
{
    int cnt = 0;
    for (auto &it : block_cache_table)
    {
        if (!it.valid)
            cnt++;
    }
    return cnt;
}

void CUDAVolumeBlockCacheImpl::clear()
{
    for (auto &it : block_cache_table)
    {
        it.valid = false;
    }
    mapping_table.assign(mapping_table.size(), 0);
}

bool CUDAVolumeBlockCacheImpl::SetCachedBlockValid(const std::array<uint32_t, 4> &target)
{
    for (auto &it : block_cache_table)
    {
        if (it.block_index == target && it.cached)
        {
            it.valid = true;
            it.t = ++count;
            this->updateMappingTable(target, it.pos_index, true);
            return true;
        }
    }
    //query chunk cache
    if(chunk_cache){
        size_t cacheID = BlockIndexToCacheID(target);
        bool cached = chunk_cache->Query(cacheID);
        if(cached){

            auto cache = chunk_cache->GetCache(cacheID);
            assert(cache.data);
            UploadVolumeBlock(target,reinterpret_cast<uint8_t*>(cache.data),cache.size,false);
            LOG_DEBUG("Upload volume block({} {} {} {}) from chunk cache",target[0],target[1],target[2],target[3]);
            return true;
        }
    }
    return false;
}

void CUDAVolumeBlockCacheImpl::SetBlockInvalid(const std::array<uint32_t, 4> &target)
{
    for (auto &it : block_cache_table)
    {
        if (it.block_index == target)
        {
            it.valid = false;
            return;
        }
    }
}

auto CUDAVolumeBlockCacheImpl::GetMappingTable() -> const std::vector<uint32_t> &
{
    return mapping_table;
}

auto CUDAVolumeBlockCacheImpl::GetCUDATextureObjects() -> std::vector<cudaTextureObject_t>
{
    return cu_textures;
}

void CUDAVolumeBlockCacheImpl::createBlockCacheTable()
{
    for (uint32_t t = 0; t < cu_array_num; t++)
    {
        for (uint32_t k = 0; k < cu_array_shape[2] / block_length; k++)
        {
            for (uint32_t j = 0; j < cu_array_shape[1] / block_length; j++)
            {
                for (uint32_t i = 0; i < cu_array_shape[0] / block_length; i++)
                {
                    BlockCacheItem item;
                    item.pos_index = {i, j, k, t};
                    item.block_index = {INVALID, INVALID, INVALID, INVALID};
                    item.valid = false;
                    item.cached = false;
                    item.t = 0;
                    block_cache_table.push_back(item);
                }
            }
        }
    }
}

auto CUDAVolumeBlockCacheImpl::GetLodMappingTableOffset() -> const std::map<uint32_t, uint32_t> &
{
    return lod_mapping_table_offset;
}

void CUDAVolumeBlockCacheImpl::updateMappingTable(const std::array<uint32_t, 4> &index,
                                                  const std::array<uint32_t, 4> &pos, bool valid)
{
    size_t flat_idx;
    try
    {
        flat_idx = ((size_t)index[2] * lod_block_dim.at(index[3])[0] * lod_block_dim.at(index[3])[1] +
                    index[1] * lod_block_dim.at(index[3])[0] + index[0]) * 4 +
                   lod_mapping_table_offset.at(index[3]);
        mapping_table.at(flat_idx + 0) = pos[0];
        mapping_table.at(flat_idx + 1) = pos[1];
        mapping_table.at(flat_idx + 2) = pos[2];
        if (valid)
            mapping_table.at(flat_idx + 3) = pos[3] | (0x00010000);
        else
            mapping_table.at(flat_idx + 3) &= 0x0000ffff;
    }
    catch (const std::exception &err)
    {
        std::cout << index[0] << " " << index[1] << " " << index[2] << " " << index[3] << "\n"
                  << pos[0] << " " << pos[1] << " " << pos[2] << " " << pos[3] << " " << flat_idx << std::endl;
        LOG_ERROR("{0}:{1}", __FUNCTION__, err.what());
        LOG_ERROR("index {0} {1} {2} {3}, pos {4} {5} {6} {7}, flag_idx {8}", index[0], index[1], index[2],
                      index[3], pos[0], pos[1], pos[2], pos[3], flat_idx);
        std::terminate();
    }
}

bool CUDAVolumeBlockCacheImpl::getCachedPos(const std::array<uint32_t, 4> &target, std::array<uint32_t, 4> &pos)
{
    block_cache_table.sort([](const BlockCacheItem &it1, const BlockCacheItem &it2) { return it1.t < it2.t; });
    for (auto &it : block_cache_table)
    {
        if (it.block_index == target && it.cached)
        {
            //assert(!it.valid);
            LOG_DEBUG("Copy CUDA device memory to CUDA Array which already stored.");
            pos = it.pos_index;
            return true;
        }
    }
    for (const auto &it : block_cache_table)
    {
        if (!it.valid && !it.cached)
        {
            pos = it.pos_index;
            return false;
        }
    }
    for (const auto &it : block_cache_table)
    {
        if (!it.valid)
        {
            pos = it.pos_index;
            return false;
        }
    }
    throw std::runtime_error("Can't find empty pos in CUDA Textures");
}

auto CUDAVolumeBlockCacheImpl::GetCacheShape() -> std::array<uint32_t, 4>
{
    return std::array<uint32_t, 4>{cu_array_num, cu_array_shape[0], cu_array_shape[1], cu_array_shape[2]};
}

CUDAVolumeBlockCacheImpl::~CUDAVolumeBlockCacheImpl()
{
    LOG_DEBUG("Call ~CUDAVolumeBlockCacheImpl destructor.");
    for (auto tex : cu_textures)
        CUDA_RUNTIME_API_CALL(cudaDestroyTextureObject(tex));
    for (auto arr : cu_arrays)
        CUDA_RUNTIME_API_CALL(cudaFreeArray(arr));
}

VS_END
