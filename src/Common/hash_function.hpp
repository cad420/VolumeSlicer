//
// Created by wyz on 2021/7/8.
//

#ifndef CGUTILS_HASH_FUNCTION_HPP
#define CGUTILS_HASH_FUNCTION_HPP

#include <cstdint>
#include <array>

using UInt32Array2=std::array<uint32_t,2>;
using UInt32Array3=std::array<uint32_t,3>;
using UInt32Array4=std::array<uint32_t,4>;

struct Hash_UInt32Array2{
    size_t operator()(const UInt32Array2& a) const {
        size_t mask=0xffffffff;
        return ((a[0]&mask)<<32)|((a[1]&mask));
    }
};
struct Hash_UInt32Array3{
    size_t operator()(const UInt32Array3& a) const {
        size_t mask=0xfffff;
        return ((a[0]&mask)<<40) | ((a[1]&mask)<<20) | (a[2]&mask);
    }
};
struct Hash_UInt32Array4{
    size_t operator()(const UInt32Array4& a) const {
        size_t mask=0xffff;
        return ((a[0]&mask)<<48)|((a[1]&mask)<<32)|((a[2]&mask)<<16)|(a[3]&mask);
    }
};

#endif //CGUTILS_HASH_FUNCTION_HPP
