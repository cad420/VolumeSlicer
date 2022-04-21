//
// Created by wyz on 2021/6/15.
//

#pragma once

#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Common/status.hpp>

#include <array>

VS_START
class Camera{
public:
    float zoom;
    std::array<float,3> pos;//measure in voxel*space
    std::array<float,3> look_at;//point pos
    std::array<float,3> up;//normalized direction
    std::array<float,3> right;
    float n,f;
};

VS_END



