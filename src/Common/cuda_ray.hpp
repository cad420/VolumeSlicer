//
// Created by wyz on 2021/9/14.
//

#pragma once

#include <VolumeSlicer/export.hpp>
#include "Algorithm/helper_math.h"
VS_START

class CUDASimpleRay{
  public:
    __device__ CUDASimpleRay(const float3& origin,const float3& direction,float t=0.f)
        :origin(origin),direction(normalize(direction)),t(t)
    {}
    float3 origin;
    float3 direction;
    float t;
};

VS_END
