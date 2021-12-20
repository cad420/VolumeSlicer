//
// Created by wyz on 2021/8/26.
//
#include <VolumeSlicer/frame.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

VS_START

namespace
{
void stbi_write_func(void *context, void *data, int size)
{
    auto image = reinterpret_cast<std::vector<uint8_t> *>(context);
    auto encoded = reinterpret_cast<uint8_t *>(data);
    for (int i = 0; i < size; i++)
    {
        image->push_back(encoded[i]);
    }
}
} // namespace

template<> void Image<Color4b>::SaveToFile(const char *file_name)
{
    stbi_write_png(file_name, width, height, 4, data, 0);
}

Img Img::encode(const uint8_t *data, uint32_t width, uint32_t height, uint8_t channels, Img::Format format,Img::Quality quality, bool flip_vertically)
{
    if (format != Img::Format::JPEG)
    {
        throw std::domain_error("only support JPEG format");
    }
    Img img;
    img.width = width;
    img.height = height;
    img.format = format;
    img.channels = channels;
    img.data.reserve(width * height * channels);
    auto quality_value = static_cast<int>(quality);
    if (flip_vertically)
    {
        stbi_flip_vertically_on_write(1);
    }
    else
    {
        stbi_flip_vertically_on_write(0);
    }
    auto res = stbi_write_jpg_to_func(stbi_write_func, reinterpret_cast<void *>(&img.data), width, height, channels, data, quality_value);
    if (res == 0)
    {
        throw std::runtime_error("encoding image failed.");
    }

    return img;
}

VS_END
