//
// Created by wyz on 2021/11/23.
//
#pragma once
#include <seria/object.hpp>
#include <array>
#include <string>
VS_START
namespace remote{

struct Volume{
    std::array<uint32_t,3> volume_dim;
    std::array<float,3> volume_space;
    std::string volume_name;

};

}

VS_END

namespace seria{
template <>
inline auto register_object<vs::remote::Volume>(){
    using Volume = vs::remote::Volume;
    return std::make_tuple(
            member("volume_dim",&Volume::volume_dim),
            member("volume_space",&Volume::volume_space),
            member("volume_name",&Volume::volume_name)
                           );
}
}
