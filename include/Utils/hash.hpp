//
// Created by wyz on 2021/9/8.
//

#pragma once
#include <array>
namespace std{
    template <size_t len>
    struct hash<array<uint32_t,len>>{
        size_t operator()(const array<uint32_t,len>& v) const{
            size_t l = 64 / len;
            size_t ans = 0;
            for(size_t i=0;i<len;i++){
                ans |= ((size_t)v[i] << l);
            }
            return ans;
        }
    };

    template <size_t len>
    struct hash<array<int,len>>{
        size_t operator()(const array<int,len>& v) const{
            size_t l = 64 / len;
            size_t ans = 0;
            for(size_t i=0;i<len;i++){
                ans |= ((size_t)v[i] << l);
            }
            return ans;
        }
    };



}

