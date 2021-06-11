//
// Created by wyz on 2021/6/8.
//

#ifndef VOLUMESLICER_READER_HPP
#define VOLUMESLICER_READER_HPP

#include<array>
#include<vector>
#include<memory>
#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>

VS_START

class VS_EXPORT Reader{
public:
    Reader()=default;

    static std::unique_ptr<Reader> CreateReader();

    virtual void AddLodData(int lod,const char* path) =0;

    virtual void GetPacket(const std::array<uint32_t,4>& idx,std::vector<std::vector<uint8_t>>& packet)=0;

    virtual size_t GetBlockSizeByte() const =0;

    virtual auto GetDim(int lod)->std::array<uint32_t,3> =0;
};



VS_END

#endif //VOLUMESLICER_READER_HPP
