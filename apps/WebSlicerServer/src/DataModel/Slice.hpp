//
// Created by wyz on 2021/10/28.
//
#pragma once
#include <VolumeSlicer/Data/slice.hpp>
#include <seria/object.hpp>
VS_START
namespace remote{

using Slice = Slice;

}

VS_END

namespace seria{
    template <>
    inline auto register_object<vs::remote::Slice>(){
        using Slice = vs::remote::Slice;
        return std::make_tuple(member("origin",&Slice::origin),
                               member("normal",&Slice::normal),
                               member("up",&Slice::up),
                               member("right",&Slice::right),
                               member("n_pixels_width",&Slice::n_pixels_width),
                               member("n_pixels_height",&Slice::n_pixels_height),
                               member("voxel_per_pixel_width",&Slice::voxel_per_pixel_width),
                               member("voxel_per_pixel_height",&Slice::voxel_per_pixel_height),
                               member("depth",&Slice::depth),
                               member("direction",&Slice::direction)
                               );
    }
}
