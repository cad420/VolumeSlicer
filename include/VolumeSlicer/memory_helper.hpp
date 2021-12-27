//
// Created by wyz on 2021/12/20.
//

#pragma once

#include <VolumeSlicer/export.hpp>
#include <VolumeSlicer/define.hpp>
#include <cstdint>
#include <cstddef>
VS_START

/**
 * @brief Some helpful functions and recommended/default settings for RTX3090
 */
struct MemoryHelper{

    static void GetGPUMemoryInfo(size_t& free, size_t& total);

    inline static int DefaultGPUTextureSizeX = 1024;

    inline static int DefaultGPUTextureSizeY = 1024;

    inline static int DefaultGPUTextureSizeZ = 1024;

    inline static int MAXGPUMemoryUsageGB = 16;

    inline static float GPUMemoryUseRatio = 0.75f;

    /**
     * @brief Get recommended GPU texture size based on type T and DefaultGPUTextureSizeXYZ.
     * @note If single texture size is too big in bytes it may create failed, so you can change DefaultGPUTextureSizeXYZ.
     */
    template <typename T>
    static void GetRecommendGPUTextureNum(int& num);


    static void GetCPUMemoryInfo(size_t& free, size_t& total);

    inline static int DefaultCPUTextureSizeX = 1024;

    inline static int DefaultCPUTextureSizeY = 1024;

    inline static int DefaultCPUTextureSizeZ = 1024;

    inline static int MAXCPUMemoryUsageGB = 24;

    inline static float CPUMemoryUseRatio = 0.75f;

    template <typename T>
    static void GetRecommendCPUTextureNum(int& num);
};



VS_END