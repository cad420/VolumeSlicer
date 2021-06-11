//
// Created by wyz on 2021/6/10.
//

#ifndef VOLUMESLICER_READER_IMPL_HPP
#define VOLUMESLICER_READER_IMPL_HPP
#include<unordered_map>
#include<VolumeSlicer/reader.hpp>
#include<VoxelCompression/voxel_compress/VoxelCmpDS.h>
VS_START

class ReaderImpl: public Reader{
public:
    ReaderImpl()=default;

    void AddLodData(int lod,const char* path) override;

    void GetPacket(const std::array<uint32_t,4>& idx,std::vector<std::vector<uint8_t>>& packet) override;

    size_t GetBlockSizeByte() const override;

    auto GetDim(int lod)->std::array<uint32_t,3> override;
private:
    std::unordered_map<int,std::unique_ptr<sv::Reader> > readers;

};

VS_END

#endif //VOLUMESLICER_READER_IMPL_HPP
