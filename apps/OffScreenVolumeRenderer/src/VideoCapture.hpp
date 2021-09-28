//
// Created by wyz on 2021/9/27.
//
#pragma once
#include <memory>
#include <cstdint>
class VideoCaptureImpl;

class VideoCapture
{
  public :

    VideoCapture(const char* filename,int width, int height,int fps);

    void AddFrame(const uint8_t* pixels);

    ~VideoCapture();

  private:
    std::unique_ptr<VideoCaptureImpl> impl;
};