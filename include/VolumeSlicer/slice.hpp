//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_SLICE_HPP
#define VOLUMESLICER_SLICE_HPP

#include<cstdint>
#include<array>
#include<memory>
#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>
VS_START


struct alignas(16) Slice{
    std::array<float,4> origin;
    std::array<float,4> normal;
    std::array<float,4> up;
    std::array<float,4> right;
    uint32_t n_pixels_width;
    uint32_t n_pixels_height;
    float voxel_per_pixel_width;
    float voxel_per_pixel_height;
};

class VS_EXPORT Slicer{
public:
    Slicer()=default;

    Slicer(const Slicer&)=delete;

    Slicer(Slicer&&)=delete;

    Slicer& operator=(const Slicer&)=delete;

    Slicer& operator=(Slicer&&)=delete;

    virtual ~Slicer(){};

    static std::unique_ptr<Slicer> CreateSlicer(const Slice& slice) noexcept;

    virtual void RotateByX(float)=0;

    virtual void RotateByY(float)=0;

    virtual void RotateByZ(float)=0;

    virtual void MoveByNormal(float dist)=0;

    virtual void MoveInPlane(float offsetX,float offsetY)=0;

    virtual void StretchInXY(float scaleX,float scaleY)=0;

    //cpu host ptr
    virtual uint8_t* GetImageData()=0;

    virtual uint32_t GetImageW() const=0;

    virtual uint32_t GetImageH() const=0;

    virtual Slice GetSlice() const=0;
};



VS_END

#endif //VOLUMESLICER_SLICE_HPP
