//
// Created by wyz on 2021/9/28.
//
#pragma once
#include "VideoCapture.hpp"

class OffScreenVolumeRenderer{
  public:
    OffScreenVolumeRenderer()=delete;
    static void RenderFrames(const char* config_file);
};