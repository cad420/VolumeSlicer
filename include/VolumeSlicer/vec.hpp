//
// Created by wyz on 2021/8/31.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <glm/glm.hpp>
VS_START

template <typename T,int len>
using Vec=glm::vec<len,T>;

template <typename T>
using Vec2=Vec<T,2>;

using Vec2i=Vec2<int>;
using Vec2f=Vec2<float>;
using Vec2d=Vec2<double>;

template <typename T>
using Vec3=Vec<T,3>;

using Vec3i=Vec3<int>;
using Vec3f=Vec3<float>;
using Vec3d=Vec3<double>;

template <typename T>
using Vec4=Vec<T,4>;

using Vec4i=Vec4<int>;
using Vec4f=Vec4<float>;
using Vec4d=Vec4<double>;



template <typename T>
T Normalize(const T& t){
    return glm::normalize(t);
}
template <typename T>
T Radians(T t){
    return glm::radians(t);
}

template <typename T,int len>
T Dot(const Vec<T,len>& v1,const Vec<T,len>& v2){
    return glm::dot(v1,v2);
}
template <typename T,int len>
T Length(const Vec<T,len>& v){
    return glm::length(v);
}


VS_END
