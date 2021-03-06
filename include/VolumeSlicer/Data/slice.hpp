//
// Created by wyz on 2021/6/7.
//

#pragma once

#include <cstdint>
#include <array>
#include <memory>

#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Common/status.hpp>

VS_START

/**
 * @brief Structure for define all properties of a 3d-slice
 */
struct alignas(16) Slice{
    std::array<float,4> origin;//measure in voxel
    std::array<float,4> normal;
    std::array<float,4> up;
    std::array<float,4> right;
    uint32_t n_pixels_width;
    uint32_t n_pixels_height;
    float voxel_per_pixel_width;
    float voxel_per_pixel_height;
    float depth = 0.f;
    [[deprecated]] int direction = 0;
    uint64_t padding = 0;
};

/**
 * @brief Class for helping operate slice like move and zoom.
 */
class VS_EXPORT Slicer: public std::enable_shared_from_this<Slicer>{
public:
    Slicer() = default;

    Slicer(const Slicer&) = delete;

    Slicer(Slicer&&) = delete;

    Slicer& operator=(const Slicer&) = delete;

    Slicer& operator=(Slicer&&) = delete;

    virtual ~Slicer(){};

    static std::unique_ptr<Slicer> CreateSlicer(const Slice& slice) noexcept;

    virtual void SetSlice(const Slice& slice) = 0;

    virtual Slice GetSlice() const = 0;

    virtual bool IsModified() const = 0;

    virtual void SetStatus(bool modified) = 0;

    virtual void SetSliceSpaceRatio(const std::array<float,3>& ratio) = 0;

    //***************************************
    //functions operate slice
    virtual void RotateByX(float) = 0;

    virtual void RotateByY(float) = 0;

    virtual void RotateByZ(float) = 0;

    virtual void NormalX() = 0;

    virtual void NormalY() = 0;

    virtual void NormalZ() = 0;

    virtual void NormalIncreaseX(float) = 0;

    virtual void NormalIncreaseY(float) = 0;

    virtual void NormalIncreaseZ(float) = 0;

    virtual void MoveByNormal(float dist) = 0;

    virtual void MoveInPlane(float offsetX,float offsetY) = 0;

    virtual void StretchInXY(float scaleX,float scaleY) = 0;

    //***************************************
    //cpu host ptr
    //image data inside slicer should set user by GetImageData

    virtual uint8_t* GetImageData() = 0;

    virtual uint32_t GetImageW() const = 0;

    virtual uint32_t GetImageH() const = 0;

    virtual void resize(int w,int h) = 0;

};



VS_END


