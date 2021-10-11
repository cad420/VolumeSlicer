//
// Created by wyz on 2021/9/14.
//

#pragma once
#include "cuda_ray.hpp"
VS_START


class CUDABox{
  public:
    __device__ __host__ CUDABox(const float3& min_p,const float3& max_p):min_p(min_p),max_p(max_p){}
    friend std::ostream& operator<<(std::ostream& os,const CUDABox& box){
        os<<"min p: ("<<box.min_p.x<<" "<<box.min_p.y<<" "<<box.min_p.z<<")\t"
          <<"max p: ("<<box.max_p.x<<" "<<box.max_p.y<<" "<<box.max_p.z<<")"<<std::endl;
        return os;
    }
    __device__ __host__ CUDABox Expand(int r) const{
        if(r<=0) return *this;
        auto d = (max_p - min_p) * (r*1.0);
        auto n_min_p = min_p - d;
        auto n_max_p = max_p + d;
        return CUDABox(n_min_p,n_max_p);
    }
    float3 min_p,max_p;
};
__device__ __host__
inline CUDABox ExpandCUDABox(int r,const float3& min_p,const float3& max_p){
    if(r <= 0) return CUDABox(min_p,max_p);
    auto d = (max_p - min_p) * (r*1.f);
    auto n_min_p = min_p - d;
    auto n_max_p = max_p + d;
    return CUDABox(n_min_p,n_max_p);
}

__device__ __host__
inline float2 IntersectWithAABB(const CUDABox& box,const CUDASimpleRay& ray){
    float t_min_x=(box.min_p.x-ray.origin.x)/ray.direction.x;
    float t_max_x=(box.max_p.x-ray.origin.x)/ray.direction.x;
    if(ray.direction.x<0.0){
        auto t  = t_min_x;
        t_min_x = t_max_x;
        t_max_x = t;
    }
    float t_min_y=(box.min_p.y-ray.origin.y)/ray.direction.y;
    float t_max_y=(box.max_p.y-ray.origin.y)/ray.direction.y;
    if(ray.direction.y<0.0){
        auto t  = t_min_y;
        t_min_y = t_max_y;
        t_max_y = t;
    }
    float t_min_z=(box.min_p.z-ray.origin.z)/ray.direction.z;
    float t_max_z=(box.max_p.z-ray.origin.z)/ray.direction.z;
    if(ray.direction.z<0.0){
        auto t  = t_min_z;
        t_min_z = t_max_z;
        t_max_z = t;
    }
    float enter_t = max(t_min_x,max(t_min_y,t_min_z));
    float exit_t  = min(t_max_x,min(t_max_y,t_max_z));
    return make_float2(enter_t,exit_t);
}





VS_END