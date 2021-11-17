//
// Created by wyz on 2021/11/1.
//

#pragma once
#include <VolumeSlicer/frame.hpp>
#include <seria/object.hpp>
VS_START
namespace remote{

using Image = Img;
}

VS_END
namespace seria{
    template <>
    inline auto register_object<vs::remote::Image>(){
        using Image = vs::remote::Image;
        return std::make_tuple(
            member("width",&Image::width),
            member("height",&Image::height),
            member("data",&Image::data)
            );
    }
}