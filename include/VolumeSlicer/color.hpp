//
// Created by wyz on 2021/8/26.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <glm/glm.hpp>
VS_START

template <typename T> using Color3 = glm::vec<3, T>;

using Color3f = Color3<float>;
using Color3d = Color3<double>;
using Color3b = Color3<uint8_t>;

template <typename T> using Color4 = glm::vec<4, T>;

using Color4f = Color4<float>;
using Color4d = Color4<double>;
using Color4b = Color4<uint8_t>;

#define DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(COLOR, OP)                                                     \
    inline COLOR operator OP(const COLOR &color, double v)                                                                    \
    {                                                                                                                  \
        COLOR n_color;                                                                                                 \
        n_color.x = color.x OP v;                                                                                       \
        n_color.y = color.y OP v;                                                                                       \
        n_color.z = color.z OP v;                                                                                       \
        return n_color;                                                                                                \
    }

DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3f, *)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3f, +)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3f, -)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3f, /)

DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3b, *)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3b, +)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3b, -)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3b, /)

DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3d, *)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3d, +)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3d, -)
DEFINE_COLOR3_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color3d, /)

#define DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(COLOR, OP)                                                     \
    inline COLOR operator OP(const COLOR &color, double v)                                                                    \
    {                                                                                                                  \
        COLOR n_color;                                                                                                 \
        n_color.x = color.x OP v;                                                                                       \
        n_color.y = color.y OP v;                                                                                       \
        n_color.z = color.z OP v;                                                                                       \
        n_color.w = color.w OP v;                                                                                       \
        return n_color;                                                                                                \
    }

DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4f, *)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4f, +)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4f, -)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4f, /)

DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4b, *)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4b, +)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4b, -)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4b, /)

DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4d, *)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4d, +)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4d, -)
DEFINE_COLOR4_UNARY_OPERATOR_WITH_DOUBLE_SCALAR(Color4d, /)

VS_END
