//
// Created by wyz on 2021/6/11.
//
#include "volume_sampler_impl.hpp"
#include "comp_volume_sample.cuh"
#include "raw_volume_sample.cuh"

#include <VolumeSlicer/Utils/logger.hpp>

VS_START

/**************************************************************************************************
 * API for RawVolumeSamplerImpl
 */

std::unique_ptr<VolumeSampler> VolumeSampler::CreateVolumeSampler(const std::shared_ptr<RawVolume> &volume)
{
    return std::make_unique<VolumeSamplerImpl<RawVolume>>(volume);
}
VolumeSamplerImpl<RawVolume>::VolumeSamplerImpl(const std::shared_ptr<RawVolume> &volume) : raw_volume(volume)
{
    this->cuda_raw_volume_sampler = std::make_unique<CUDARawVolumeSampler>(GetCUDACtx());
    this->cuda_raw_volume_sampler->SetVolumeData(raw_volume->GetData(), raw_volume->GetVolumeDimX(),
                                                 raw_volume->GetVolumeDimY(), raw_volume->GetVolumeDimZ());
    LOG_INFO("Successfully create raw volume sampler.");
}

bool VolumeSamplerImpl<RawVolume>::Sample(const Slice &slice, uint8_t *data, bool)
{
    cuda_raw_volume_sampler->Sample(data, slice, raw_volume->GetVolumeSpaceX(), raw_volume->GetVolumeSpaceY(),
                                    raw_volume->GetVolumeSpaceZ());
    return true;
}

VolumeSamplerImpl<RawVolume>::~VolumeSamplerImpl()
{
    LOG_INFO("Call ~VolumeSamplerImpl<RawVolume> destructor.");
    raw_volume.reset();
}

/**************************************************************************************************
 * API for CompVolumeSamplerImpl
 */

std::unique_ptr<VolumeSampler> VolumeSampler::CreateVolumeSampler(const std::shared_ptr<CompVolume> &volume)
{
    return std::make_unique<VolumeSamplerImpl<CompVolume>>(volume);
}

VolumeSamplerImpl<CompVolume>::VolumeSamplerImpl(const std::shared_ptr<CompVolume> &volume) : comp_volume(volume)
{
    initVolumeInfo();
    this->cuda_volume_block_cache = CUDAVolumeBlockCache::Create();
    this->cuda_volume_block_cache->SetCacheBlockLength(comp_volume->GetBlockLength()[0]);
    this->cuda_volume_block_cache->SetCacheCapacity(4, 2048, 1024, 1024);
    this->cuda_volume_block_cache->CreateMappingTable(this->comp_volume->GetBlockDim());
    this->cuda_comp_volume_sampler = std::make_unique<CUDACompVolumeSampler>(GetCUDACtx());
    this->cuda_comp_volume_sampler->SetBlockInfo(block_length, padding);
    {
        auto &mapping_table = this->cuda_volume_block_cache->GetMappingTable();
        this->cuda_comp_volume_sampler->UploadMappingTable(mapping_table.data(), mapping_table.size());
        auto &lod_mapping_table_offset = this->cuda_volume_block_cache->GetLodMappingTableOffset();
        std::vector<uint32_t> offset; // for one block not for uint32_t
        offset.resize(max_lod + 1, 0);
        for (auto &it : lod_mapping_table_offset)
        {
            if (it.first <= max_lod)
                offset.at(it.first) = it.second / 4;
        }
        this->cuda_comp_volume_sampler->UploadLodMappingTableOffset(offset.data(), offset.size());
        auto texes = this->cuda_volume_block_cache->GetCUDATextureObjects();
        this->cuda_comp_volume_sampler->SetCUDATextureObject(texes.data(), texes.size());
    }

    BlockParameter block_parameter;
    block_parameter.block_length = this->block_length;
    block_parameter.padding = this->padding;
    block_parameter.no_padding_block_length = this->no_padding_block_length;
    block_parameter.block_dim =
        make_int3(this->lod_block_dim.at(0)[0], this->lod_block_dim.at(0)[1], this->lod_block_dim.at(0)[2]);
    auto tex_shape = this->cuda_volume_block_cache->GetCacheShape();
    block_parameter.texture_size3 = make_int3(tex_shape[1], tex_shape[2], tex_shape[3]);
    cuda_comp_volume_sampler->UploadBlockParameter(block_parameter);
}

/**
 * to dynamic change space in runtime, space should only get by GetVolumeSpace only in this Sample function.
 */
bool VolumeSamplerImpl<CompVolume>::Sample(const Slice &slice, uint8_t *data, bool async)
{

    glm::vec3 origin = {slice.origin[0], slice.origin[1], slice.origin[2]};
    glm::vec3 right = {slice.right[0], slice.right[1], slice.right[2]};
    glm::vec3 up = {slice.up[0], slice.up[1], slice.up[2]};
    glm::vec3 normal = {slice.normal[0], slice.normal[1], slice.normal[2]};
    auto old_right = right;
    auto old_up = up;

    glm::vec3 space = {comp_volume->GetVolumeSpaceX(), comp_volume->GetVolumeSpaceY(), comp_volume->GetVolumeSpaceZ()};
    float base_space = std::min({space.x, space.y, space.z});
    auto old_origin = origin;
    origin = origin;
    // transform right and up to voxel-coord and then get slice's obb's right and up
    // set slice's obb's normal length to 1 or 2*padding
    right = right * (slice.n_pixels_width * slice.voxel_per_pixel_width) * base_space / space / 2.f;
    up = up * (slice.n_pixels_height * slice.voxel_per_pixel_height) * base_space / space / 2.f;
    normal = glm::normalize(normal); // for preload
    OBB obb(origin, right, up, normal);

    assert(slice.voxel_per_pixel_width == slice.voxel_per_pixel_height);
    this->current_lod = evaluateLod(slice.voxel_per_pixel_width);

    CompSampleParameter comp_sampler_parameter;
    comp_sampler_parameter.image_w = slice.n_pixels_width;
    comp_sampler_parameter.image_h = slice.n_pixels_height;
    comp_sampler_parameter.lod = this->current_lod;
    right = glm::normalize(old_right);
    up = glm::normalize(old_up);
    comp_sampler_parameter.origin = make_float3(old_origin.x, old_origin.y, old_origin.z);
    comp_sampler_parameter.right = make_float3(right.x, right.y, right.z);
    comp_sampler_parameter.down = make_float3(-up.x, -up.y, -up.z);
    comp_sampler_parameter.space = make_float3(space.x, space.y, space.z);
    comp_sampler_parameter.space_ratio = make_float3(space.x / base_space, space.y / base_space, space.z / base_space);
    comp_sampler_parameter.voxels_per_pixel = make_float2(slice.voxel_per_pixel_width, slice.voxel_per_pixel_height);
    //params for slice that has depth attribute
    {
        comp_sampler_parameter.normal = make_float3(normal.x,normal.y,normal.z);
        float sample_space = base_space * 0.5f;
        int steps = std::ceil(slice.depth / sample_space );
        comp_sampler_parameter.step = steps==0 ? 0 : slice.depth / steps;
        LOG_INFO("sample steps {}",steps);
        comp_sampler_parameter.steps = steps;
        comp_sampler_parameter.direction = slice.direction;
    }
    cuda_comp_volume_sampler->UploadCompSampleParameter(comp_sampler_parameter);

    calcIntersectBlocks(obb);

    filterIntersectBlocks();

    sendRequests();

    do
    {
        fetchBlocks(async);
    } while (!async && !isSampleComplete());

    clearCurrentInfo();

    auto &mapping_table = this->cuda_volume_block_cache->GetMappingTable();
    this->cuda_comp_volume_sampler->UploadMappingTable(mapping_table.data(), mapping_table.size());

    cuda_comp_volume_sampler->Sample(data, slice);

    return isSampleComplete();
}

void VolumeSamplerImpl<CompVolume>::initVolumeInfo()
{
    auto l = comp_volume->GetBlockLength();
    this->block_length = l[0];
    this->padding = l[1];
    this->no_padding_block_length = block_length - 2 * padding;
    this->min_lod = l[2];
    this->max_lod = l[3];
    LOG_INFO("VolumeSampler: block_length({0}), padding({1}), min_lod({2}), max_lod({3}).", block_length, padding,
             min_lod, max_lod);
    for (int i = min_lod; i <= max_lod; i++)
    {
        lod_block_dim[i] = comp_volume->GetBlockDim(i);
        LOG_INFO("VolumeSampler: lod({0})'s dim ({1},{2},{3}).", i, lod_block_dim.at(i)[0], lod_block_dim.at(i)[1],
                 lod_block_dim.at(i)[2]);
    }
    createVirtualBlocks();
}

void VolumeSamplerImpl<CompVolume>::createVirtualBlocks()
{
    for (uint32_t i = min_lod; i <= max_lod; i++)
    {
        uint32_t t = std::pow(2, i - min_lod);
        auto dim_x = lod_block_dim.at(i)[0];
        auto dim_y = lod_block_dim.at(i)[1];
        auto dim_z = lod_block_dim.at(i)[2];
        uint32_t lod_no_padding_block_length = no_padding_block_length * t;
        for (uint32_t z = 0; z < dim_z; z++)
        {
            for (uint32_t y = 0; y < dim_y; y++)
            {
                for (uint32_t x = 0; x < dim_x; x++)
                {
                    virtual_blocks[i].emplace_back(
                        glm::vec3{x * lod_no_padding_block_length, y * lod_no_padding_block_length,
                                  z * lod_no_padding_block_length},
                        glm::vec3{(x + 1) * lod_no_padding_block_length, (y + 1) * lod_no_padding_block_length,
                                  (z + 1) * lod_no_padding_block_length},
                        std::array<uint32_t, 4>{x, y, z, i});
                }
            }
        }
    }
}

VolumeSamplerImpl<CompVolume>::~VolumeSamplerImpl()
{
    LOG_INFO("Deleting CompVolumeSampler...");
    // not call is also ok, just need base class's virtual destruct function be called,
    // if that called, compiler will bind the real class to ptr and call this destruct function,
    // comp_volume would auto destruct
    comp_volume.reset();
    LOG_INFO("Deleted CompVolumeSampler...");
}

void VolumeSamplerImpl<CompVolume>::sendRequests()
{

    comp_volume->PauseLoadBlock(); // not necessary

    /**
     * not used if ClearBlockInQueue use current_intersect_blocks
     * fix bug if use new_need_blocks because it would clear new need blocks last time which are not new_need_blocks now
    if(comp_volume->GetStatus()){
        auto un_upload_blocks=cuda_comp_volume_sampler->GetUnUploadBlocks();
        for(auto& it:un_upload_blocks){
            spdlog::critical("{0} {1} {2} {3}.",it[0],it[1],it[2],it[3]);

            if(current_intersect_blocks.find(it)!=current_intersect_blocks.end()){
                spdlog::critical("insert!!!");
                new_need_blocks.insert(it);
            }
        }
    }
     */

    if (!current_intersect_blocks.empty())
    {
        std::vector<std::array<uint32_t, 4>> targets;
        targets.reserve(current_intersect_blocks.size());
        for (auto &it : current_intersect_blocks)
            targets.push_back(it.index);
        // may clear new need last time if use new_need_blocks
        comp_volume->ClearBlockInQueue(targets);
        //        comp_volume->ClearBlockQueue();
    }

    for (auto &it : new_need_blocks)
    {
        LOG_INFO("send {0} {1} {2} {3}.", it.index[0], it.index[1], it.index[2], it.index[3]);
        comp_volume->SetRequestBlock(it.index);
    }
    //    new_need_blocks.clear();

    for (auto &it : no_need_blocks)
    {
        comp_volume->EraseBlockInRequest(it.index);
        LOG_INFO("erase {0} {1} {2} {3}.", it.index[0], it.index[1], it.index[2], it.index[3]);
    }
    //    no_need_blocks.clear();

    comp_volume->StartLoadBlock(); // not necessary
}
void VolumeSamplerImpl<CompVolume>::clearCurrentInfo()
{
    new_need_blocks.clear();
    no_need_blocks.clear();
}
void VolumeSamplerImpl<CompVolume>::fetchBlocks(bool async)
{
    this->is_sample_complete = true;
    if (!async)
    {
        auto tmp = new_need_blocks;
        for (auto &it : tmp)
        {
            auto block = comp_volume->GetBlock(it.index);
            if (block.valid)
            {
                assert(block.block_data->GetDataPtr());
                this->cuda_volume_block_cache->UploadVolumeBlock(block.index, block.block_data->GetDataPtr(),
                                                                 block.block_data->GetSize(), true);
                block.Release();
                new_need_blocks.erase(it);
            }
            else
            {
                this->is_sample_complete = false;
            }
        }
    }
    else
    {
        for (auto &it : current_intersect_blocks)
        {
            auto block = comp_volume->GetBlock(it.index);
            if (block.valid)
            {
                assert(block.block_data->GetDataPtr());
                this->cuda_volume_block_cache->UploadVolumeBlock(block.index, block.block_data->GetDataPtr(),
                                                                 block.block_data->GetSize(), true);
                block.Release();
            }
            else
            {
                this->is_sample_complete = false;
            }
        }
    }
}

void VolumeSamplerImpl<CompVolume>::filterIntersectBlocks()
{

    if (!new_need_blocks.empty())
    {
        std::unordered_set<AABB, AABBHash> temp;
        for (const auto &it : new_need_blocks)
        {
            bool cached = this->cuda_volume_block_cache->SetCachedBlockValid(it.index);

            if (cached)
            {
                //            new_need_blocks.erase(it);
            }
            else
            {
                temp.insert(it);
            }
        }
        new_need_blocks = std::move(temp);
    }

    if (!no_need_blocks.empty())
    {
        for (auto &it : no_need_blocks)
        {
            this->cuda_volume_block_cache->SetBlockInvalid(it.index);
        }
    }

}

uint32_t VolumeSamplerImpl<CompVolume>::evaluateLod(float voxels_per_pixel)
{
    if (voxels_per_pixel < 1.f)
        return 0;
    auto res = std::log2f(voxels_per_pixel);
    return res > max_lod ? max_lod : res;
}

void VolumeSamplerImpl<CompVolume>::calcIntersectBlocks(const OBB &obb)
{
    auto aabb = obb.getAABB();
    std::unordered_set<AABB, AABBHash> current_aabb_intersect_blocks;
    std::unordered_set<AABB, AABBHash> current_obb_intersect_blocks;

    for (auto &it : virtual_blocks.at(this->current_lod))
    {
        if (aabb.intersect(it))
        {
            current_aabb_intersect_blocks.insert(it);
        }
    }

    for (auto &it : current_aabb_intersect_blocks)
    {
        if (obb.intersect_obb(it.convertToOBB()))
        {
            current_obb_intersect_blocks.insert(it);
        }
    }

    for (auto &it : current_obb_intersect_blocks)
    {
        if (current_intersect_blocks.find(it) == current_intersect_blocks.end())
        {
            new_need_blocks.insert(it);
        }
    }

    for (auto &it : current_intersect_blocks)
    {
        if (current_obb_intersect_blocks.find(it) == current_obb_intersect_blocks.end())
        {
            no_need_blocks.insert(it);
        }
    }

    current_intersect_blocks = std::move(current_obb_intersect_blocks);
}

bool VolumeSamplerImpl<CompVolume>::isSampleComplete() const
{
    return is_sample_complete;
}

VS_END
