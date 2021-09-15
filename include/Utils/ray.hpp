//
// Created by wyz on 2021/9/7.
//

#ifndef VOLUMESLICER_RAY_HPP
#define VOLUMESLICER_RAY_HPP
#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/vec.hpp>
VS_START

class SimpleRay{
  public:
    __device__ __host__ SimpleRay(const Vec3d& origin,const Vec3d& direction,double t=0.0)
        :origin(origin),direction(Normalize(direction)),t(t)
    {}
    Vec3d origin;
    Vec3d direction;
    double t;
};

VS_END
#endif // VOLUMESLICER_RAY_HPP
