//
// Created by wyz on 2021/8/26.
//

#pragma once

#include <VolumeSlicer/Common/color.hpp>
#include <VolumeSlicer/Utils/block_array.hpp>
#include <VolumeSlicer/Utils/math.hpp>

#include "texture.hpp"

VS_START

class LinearSampler
{
  public:
    template <typename Texel>
    static auto Sample1D(const Texture1D<Texel> &tex, double u) -> Texel
    {
        u = Clamp(u, 0.0, 1.0) * (tex.GetLength() - 1);
        int u0 = Clamp(static_cast<int>(u), 0, static_cast<int>(tex.GetLength() - 1));
        int u1 = Clamp(u0 + 1, 0, static_cast<int>(tex.GetLength() - 1));
        double d_u = u - u0;
        return tex(u0) * (1 - d_u) + tex(u1) * d_u;
    }

    template <typename Texel>
    static auto Sample2D(const Texture2D<Texel> &tex, double u, double v) -> Texel
    {
        u = Clamp(u, 0.0, 1.0) * (tex.GetWidth() - 1);
        v = Clamp(v, 0.0, 1.0) * (tex.GetHeight() - 1);
        int u0 = Clamp(static_cast<int>(u), 0, static_cast<int>(tex.GetWidth() - 1));
        int u1 = Clamp(u0 + 1, 0, static_cast<int>(tex.GetWidth() - 1));
        int v0 = Clamp(static_cast<int>(v), 0, static_cast<int>(tex.GetHeight() - 1));
        int v1 = Clamp(v0 + 1, 0, static_cast<int>(tex.GetHeight() - 1));
        double d_u = u - u0;
        double d_v = v - v0;
        return (tex(u0, v0) * (1.0 - d_u) + tex(u1, v0) * d_u) * (1.0 - d_v) +
               (tex(u0, v1) * (1.0 - d_u) + tex(u1, v1) * d_u) * d_v;
    }

    template <typename Texel>
    static auto Sample3D(const Texture3D<Texel> &tex, double u, double v, double k) -> Texel
    {
        u = Clamp(u, 0.0, 1.0) * (tex.GetXSize() - 1);
        v = Clamp(v, 0.0, 1.0) * (tex.GetYSize() - 1);
        k = Clamp(k, 0.0, 1.0) * (tex.GetZSize() - 1);
        int u0 = Clamp(static_cast<int>(u), 0, static_cast<int>(tex.GetXSize() - 1));
        int u1 = Clamp(u0 + 1, 0, static_cast<int>(tex.GetXSize() - 1));
        int v0 = Clamp(static_cast<int>(v), 0, static_cast<int>(tex.GetYSize() - 1));
        int v1 = Clamp(v0 + 1, 0, static_cast<int>(tex.GetYSize() - 1));
        int k0 = Clamp(static_cast<int>(k), 0, static_cast<int>(tex.GetZSize() - 1));
        int k1 = Clamp(k0 + 1, 0, static_cast<int>(tex.GetZSize() - 1));
        double d_u = u - u0;
        double d_v = v - v0;
        double d_k = k - k0;
        return ((tex(u0, v0, k0) * (1.0 - d_u) + tex(u1, v0, k0) * d_u) * (1.0 - d_v) +
                (tex(u0, v1, k0) * (1.0 - d_u) + tex(u1, v1, k0) * d_u) * d_v) *
                   (1.0 - d_k) +
               ((tex(u0, v0, k1) * (1.0 - d_u) + tex(u1, v0, k1) * d_u) * (1.0 - d_v) +
                (tex(u0, v1, k1) * (1.0 - d_u) + tex(u1, v1, k1) * d_u) * d_v) *
                   d_k;
    }

    template <typename T, uint32_t nLogBlockLength>
    static auto Sample3D(const Block3DArray<T, nLogBlockLength> &block3d_array, double u, double v, double k) -> T
    {
        return block3d_array.Sample(u, v, k);
    }
};

VS_END