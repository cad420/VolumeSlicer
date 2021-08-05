//
// Created by wyz on 2021/6/11.
//

#ifndef VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP
#define VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP

#include <VolumeSlicer/volume_sampler.hpp>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include "Common/boundingbox.hpp"
#include <VolumeSlicer/volume_cache.hpp>
VS_START

template<class T>
class VolumeSamplerImpl;

class CUDARawVolumeSampler;

template<>
class VolumeSamplerImpl<RawVolume>: public VolumeSampler{
public:
    explicit VolumeSamplerImpl(const std::shared_ptr<RawVolume>& volume);

    ~VolumeSamplerImpl() override;

    bool Sample(const Slice& slice,uint8_t* data) override;

private:

private:
    std::shared_ptr<RawVolume> raw_volume;
    std::unique_ptr<CUDARawVolumeSampler> cuda_raw_volume_sampler;
};




struct AABBHash{
    std::size_t operator()(const AABB& aabb) const {
        return (aabb.index[3]<<24)+(aabb.index[0]<<16)+(aabb.index[1]<<8)+aabb.index[2];
    }
};


class CUDACompVolumeSampler;

/**
 * should not associate with CUDA or OpenGL
 */
template<>
class VolumeSamplerImpl<CompVolume>: public VolumeSampler{
public:
    explicit VolumeSamplerImpl(const std::shared_ptr<CompVolume>& volume);

    ~VolumeSamplerImpl() override;
    //data is host ptr
    bool Sample(const Slice& slice,uint8_t* data) override;

private:
    //set comp_volume's information for member data
    void initVolumeInfo();

    //according to slice's voxels_per_pixel to evaluate current lod
    uint32_t evaluateLod(float voxels_per_pixel);

    //according to current lod, calculate intersected blocks
    void calcIntersectBlocks(const OBB& obb);

    //1.remove cached blocks in new_need_blocks.
    //2.set no_need_blocks invalid for cuda_comp_volume_sampler.
    void filterIntersectBlocks();

    //send blocks' information which are current need and not need any more
    void sendRequests();

    //get blocks from comp_volume which are current need
    void fetchBlocks();
private:
    void createVirtualBlocks();
    bool isSampleComplete() const;
private:
    std::shared_ptr<CompVolume> comp_volume;
    std::unique_ptr<CUDACompVolumeSampler> cuda_comp_volume_sampler;
    bool is_sample_complete;

    uint32_t block_length,padding,no_padding_block_length;
    uint32_t min_lod,max_lod;
    std::map<uint32_t,std::array<uint32_t,3>> lod_block_dim;

    uint32_t current_lod;
    std::unordered_set<AABB,AABBHash> current_intersect_blocks;
    std::unordered_set<AABB,AABBHash> new_need_blocks,no_need_blocks;

    std::unordered_map<uint32_t,std::vector<AABB>> virtual_blocks;

    std::unique_ptr<CUDAVolumeBlockCache> cuda_volume_block_cache;
};





VS_END


#endif //VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP
