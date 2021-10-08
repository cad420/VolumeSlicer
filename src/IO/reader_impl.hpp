//
// Created by wyz on 2021/6/10.
//

#ifndef VOLUMESLICER_READER_IMPL_HPP
#define VOLUMESLICER_READER_IMPL_HPP
#include<unordered_map>
#include<VolumeSlicer/reader.hpp>
#include<VoxelCompression/voxel_compress/VoxelCmpDS.h>
#include<array>
#include <VolumeSlicer/LRU.hpp>
#include <Utils/hash.hpp>
VS_START

class ReaderImpl: public Reader{
public:
    ReaderImpl();

    void AddLodData(int lod,const char* path) override;

    void GetPacket(const std::array<uint32_t,4>& idx,std::vector<std::vector<uint8_t>>& packet) override;

    size_t GetBlockSizeByte() override;

    auto GetBlockLength() const -> std::array<uint32_t,4> override;

    auto GetBlockDim(int lod) const ->std::array<uint32_t,3> override;

    auto GetFrameShape() const ->std::array<uint32_t,2> override;
private:
    std::unordered_map<int,std::unique_ptr<sv::Reader> > readers;
    int min_lod,max_lod;
    LRUCache<std::array<uint32_t,4>,std::vector<std::vector<uint8_t>>> packet_cache;
    std::mutex mtx;
};

VS_END

#endif //VOLUMESLICER_READER_IMPL_HPP
