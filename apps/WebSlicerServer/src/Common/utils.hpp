//
// Created by wyz on 2021/10/28.
//
#pragma once
#include <string>

inline std::string URIJoin(const std::string& path){
    return path;
}
template<class... Args>
inline std::string URIJoin(const std::string& path,Args&&... res){
    return path+"/"+URIJoin(res...);
}