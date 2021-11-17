//
// Created by wyz on 2021/10/28.
//
#pragma once

#include "SliceService.hpp"
#include <memory>
#include <string>
VS_START
namespace remote{

inline auto CreateServiceByName(const std::string& name){
    if(name == "slice"){
        return std::unique_ptr<JsonRPCService>(new SliceService());
    }

}

}

VS_END