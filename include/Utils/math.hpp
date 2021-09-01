//
// Created by wyz on 2021/8/26.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <cassert>
VS_START

template <typename T>
constexpr const T& Clamp(const T& v,const T& lo,const T& hi){
    assert(!(lo>hi));
    return v<lo?lo:(v>hi?hi:v);
}



VS_END
