//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_SLICE_IMPL_HPP
#define VOLUMESLICER_SLICE_IMPL_HPP

#include<vector>
#include<glm/glm.hpp>

#include<VolumeSlicer/slice.hpp>


VS_START

class SlicerImpl: public Slicer{
public:
    explicit SlicerImpl(const Slice&);

    void SetSlice(const Slice& slice) override;

    bool IsModified() const override;

    void SetStatus(bool modified) override;

    void SetSliceSpaceRatio(const std::array<float,3>& ratio) override;

    void RotateByX(float degree) override;

    void RotateByY(float degree) override;

    void RotateByZ(float degree) override;

    void MoveByNormal(float dist) override;

    void MoveInPlane(float offsetX,float offsetY) override;

    void StretchInXY(float scaleX,float scaleY) override;

    uint8_t* GetImageData() override;

    uint32_t GetImageW() const override;

    uint32_t GetImageH() const override;

    Slice GetSlice() const override;

public:
    bool IsValidSlice(const Slice& slice) const;

private:
    bool IsValidVector() const;
private:
    glm::vec3 origin;
    glm::vec3 normal;
    glm::vec3 up;
    glm::vec3 right;
    uint32_t n_pixels_width,n_pixels_height;
    float voxel_per_pixel_width,voxel_per_pixel_height;

    std::vector<uint8_t> image;
    bool is_modified;
    glm::vec3 ratio;
};

VS_END

#endif //VOLUMESLICER_SLICE_IMPL_HPP
