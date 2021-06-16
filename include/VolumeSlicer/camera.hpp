//
// Created by wyz on 2021/6/15.
//

#ifndef VOLUMESLICER_CAMERA_HPP
#define VOLUMESLICER_CAMERA_HPP

#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>

#include<array>

VS_START
class Camera{
public:
    std::array<float,3> pos;
    std::array<float,3> front;//normalized direction
    std::array<float,3> up;//normalized direction
    float zoom;
    float n,f;
};

VS_END


#endif //VOLUMESLICER_CAMERA_HPP
