//
// Created by wyz on 2021/7/21.
//

#ifndef VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
#define VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/volume_cache.hpp>
#include <unordered_set>
#include "Common/hash_function.hpp"
VS_START
class CUDACompVolumeRendererImpl: public CUDACompVolumeRenderer{
public:
    CUDACompVolumeRendererImpl(int w,int h,CUcontext ctx);

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetMPIViewOffset(float x_offset,float y_offset) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunc(TransferFunc tf) override;

    void render() override;

    auto GetFrame()->const Image<uint32_t>& override;

    void resize(int w,int h) override;

    void clear() override;

private:
    void calcMissedBlocks();

    void filterMissedBlocks();

    void sendRequests();

    void fetchBlocks();

private:
    int window_w,window_h;
    CUcontext cu_context;
    std::shared_ptr<CompVolume> comp_volume;
    Camera camera;
    Image<uint32_t> image;
    std::unique_ptr<CUDAVolumeBlockCache> cuda_volume_block_cache;

    std::vector<uint32_t> missed_blocks_pool;
    std::unordered_set<std::array<uint32_t,4>,Hash_UInt32Array4> missed_blocks;
    std::unordered_set<std::array<uint32_t,4>,Hash_UInt32Array4> new_missed_blocks,no_missed_blocks;
    std::vector<uint32_t> block_offset;
};


VS_END
#endif //VOLUMESLICER_CUDA_COMP_RENDER_IMPL_HPP
