//
// Created by wyz on 2021/6/8.
//

#pragma once

#include <array>
#include <vector>
#include <memory>

#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Common/status.hpp>

VS_START

class VS_EXPORT Reader{
public:
    Reader()=default;

    static std::unique_ptr<Reader> CreateReader(const char* file_name);

    virtual void AddLodData(int lod,const char* path) = 0;

    //packets are encoded volume data
    virtual void GetPacket(const std::array<uint32_t,4>& idx,std::vector<std::vector<uint8_t>>& packet) = 0;

    virtual size_t GetBlockSizeByte() = 0;

    //return {block_length,padding,min_lod,max_lod}
    virtual auto GetBlockLength() const -> std::array<uint32_t,4> = 0;

    virtual auto GetBlockDim(int lod) const ->std::array<uint32_t,3> = 0;

    /**
     * @brief get decode frame size(w x h), this should be useless.
     */
    [[deprecated]] virtual auto GetFrameShape() const ->std::array<uint32_t,2> = 0;

    virtual auto GetVolumeSpace() const -> std::array<float,3> = 0;

    /**
     * @note volume space will read from file now.
     */
    [[deprecated]] virtual void SetVolumeSpace(const std::array<float,3>&) = 0;
};



VS_END


