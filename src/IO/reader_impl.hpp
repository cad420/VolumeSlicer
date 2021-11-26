//
// Created by wyz on 2021/6/10.
//

#pragma once

#include <VolumeSlicer/LRU.hpp>
#include <VolumeSlicer/Utils/hash.hpp>
#include <VolumeSlicer/reader.hpp>

#include <VoxelCompression/voxel_compress/VoxelCmpDS.h>

#include <array>
#include <unordered_map>

VS_START

class ReaderImpl: public Reader{
public:
    ReaderImpl();

    void AddLodData(int lod,const char* path) override;

    void GetPacket(const std::array<uint32_t,4>& idx,std::vector<std::vector<uint8_t>>& packet) override;

    size_t GetBlockSizeByte() override;

    auto GetBlockLength() const -> std::array<uint32_t,4> override;

    auto GetBlockDim(int lod) const ->std::array<uint32_t,3> override;

    auto GetVolumeSpace() const -> std::array<float,3> override;

    auto GetFrameShape() const ->std::array<uint32_t,2> override;

    void SetVolumeSpace(const std::array<float,3>&) override;
private:
    std::unordered_map<int,std::unique_ptr<sv::Reader> > readers;
    int min_lod,max_lod;
    std::array<float,3> volume_space;
    LRUCache<std::array<uint32_t,4>,std::vector<std::vector<uint8_t>>> packet_cache;
    std::mutex mtx;
};

VS_END
