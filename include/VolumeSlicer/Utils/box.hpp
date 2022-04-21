//
// Created by wyz on 2021/9/7.
//
#pragma once

#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Utils/ray.hpp>

#include <iostream>
#include <algorithm>
VS_START

class Box
{
  public:
    Box(const Vec3d &min_p, const Vec3d &max_p) : min_p(min_p), max_p(max_p)
    {
    }
    Box(const Box &) = default;
    Box &operator=(const Box &) = default;
    friend std::ostream &operator<<(std::ostream &os, const Box &box)
    {
        os << "min p: (" << box.min_p.x << " " << box.min_p.y << " " << box.min_p.z << ")\t"
           << "max p: (" << box.max_p.x << " " << box.max_p.y << " " << box.max_p.z << ")" << std::endl;
        return os;
    }
    Box Expand(int r) const
    {
        if (r <= 0)
            return *this;
        auto d = (max_p - min_p) * (r * 1.0);
        auto n_min_p = min_p - d;
        auto n_max_p = max_p + d;
        return Box(n_min_p, n_max_p);
    }
    Vec3d min_p, max_p;
};

inline Box ExpandBox(int r, const Vec3d &min_p, const Vec3d &max_p)
{
    if (r <= 0)
        return Box(min_p, max_p);
    auto d = (max_p - min_p) * (r * 1.0);
    auto n_min_p = min_p - d;
    auto n_max_p = max_p + d;
    return Box(n_min_p, n_max_p);
}

inline Vec2d IntersectWithAABB(const Box &box, const SimpleRay &ray)
{
    double t_min_x = (box.min_p.x - ray.origin.x) / ray.direction.x;
    double t_max_x = (box.max_p.x - ray.origin.x) / ray.direction.x;
    if (ray.direction.x < 0.0)
    {
        std::swap(t_min_x, t_max_x);
    }
    double t_min_y = (box.min_p.y - ray.origin.y) / ray.direction.y;
    double t_max_y = (box.max_p.y - ray.origin.y) / ray.direction.y;
    if (ray.direction.y < 0.0)
    {
        std::swap(t_min_y, t_max_y);
    }
    double t_min_z = (box.min_p.z - ray.origin.z) / ray.direction.z;
    double t_max_z = (box.max_p.z - ray.origin.z) / ray.direction.z;
    if (ray.direction.z < 0.0)
    {
        std::swap(t_min_z, t_max_z);
    }
    double enter_t = (std::max)(std::initializer_list<double>{t_min_x, t_min_y, t_min_z});
    double exit_t = (std::min)(std::initializer_list<double>{t_max_x, t_max_y, t_max_z});
    return {enter_t, exit_t};
}

__device__ __host__ inline bool IsIntersected(double enter_t, double exit_t)
{
    if (exit_t >= 0 && enter_t < exit_t)
        return true;
    else
        return false;
}

__device__ __host__ inline bool IsIntersected(float enter_t, float exit_t)
{
    if (exit_t >= 0 && enter_t < exit_t)
        return true;
    else
        return false;
}

VS_END
