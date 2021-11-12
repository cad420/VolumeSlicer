//
// Created by wyz on 2021/7/21.
//

#pragma once

#include <unordered_set>

#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/volume_cache.hpp>
#include <VolumeSlicer/cdf.hpp>

#include "Common/hash_function.hpp"

VS_START

class CUDACompVolumeRendererImpl: public CUDACompVolumeRenderer{
public:
    CUDACompVolumeRendererImpl(int w,int h,CUcontext ctx);

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetRenderPolicy(CompRenderPolicy) override;

    auto GetBackendName()-> std::string override;

    void SetMPIRender(MPIRenderParameter) override ;

    void SetStep(double step,int steps) override;

    void SetCamera(Camera camera) override;

    void SetTransferFunc(TransferFunc tf) override;

    void render(bool sync) override;

    auto GetImage()->const Image<Color4b>& override;

    void resize(int w,int h) override;

    void clear() override;

private:
    void calcMissedBlocks();

    void filterMissedBlocks();

    void sendRequests();

    void fetchBlocks(bool sync);

    bool isRenderFinish();

    void clearCurrentInfo();

private:
    bool is_render_finish;
    int window_w,window_h;
    CUcontext cu_context;
    std::shared_ptr<CompVolume> comp_volume;
    float step;
    int steps;
    bool mpi_render;
    Camera camera;
    Image<Color4b> image;
    std::unique_ptr<CUDAVolumeBlockCache> cuda_volume_block_cache;

    std::vector<uint32_t> missed_blocks_pool;
    std::unordered_set<std::array<uint32_t,4>,Hash_UInt32Array4> missed_blocks;
    std::unordered_set<std::array<uint32_t,4>,Hash_UInt32Array4> new_missed_blocks,no_missed_blocks;
    std::vector<uint32_t> block_offset;


    std::unique_ptr<CDFManager> cdf_manager;
    int cdf_block_length;
    int cdf_dim_x,cdf_dim_y,cdf_dim_z;
    std::unordered_map<int,std::vector<uint32_t>> volume_value_map;
};


VS_END

