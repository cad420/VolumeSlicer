//
// Created by wyz on 2021/9/8.
//

#pragma once
#include <array>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
namespace std{
    template <size_t len>
    struct hash<array<uint32_t,len>>{
        size_t operator()(const array<uint32_t,len>& v) const{
            glm::vec<len,uint32_t> t;
            for(size_t i=0;i<len;i++){
                t[i]=v[i];
            }
            return hash<glm::vec<len,uint32_t>>()(t);
        }
    };

    template <size_t len>
    struct hash<array<int,len>>{
        size_t operator()(const array<int,len>& v) const{
            glm::vec<len,int> t;
            for(size_t i=0;i<len;i++){
                t[i]=v[i];
            }
            return hash<glm::vec<len,int>>()(t);
        }
    };



}

