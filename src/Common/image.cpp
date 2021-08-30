//
// Created by wyz on 2021/8/26.
//
#include <VolumeSlicer/frame.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

VS_START

void Image<Color4b>::SaveToFile(const char *file_name)
{
    stbi_write_png(file_name,width,height,4,data,0);
}



VS_END