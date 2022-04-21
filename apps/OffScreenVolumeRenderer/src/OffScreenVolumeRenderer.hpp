//
// Created by wyz on 2021/9/28.
//
#pragma once
#include "VideoCapture.hpp"
#include <VolumeSlicer/Render/camera.hpp>
#include <VolumeSlicer/Render/transfer_function.hpp>
#include <functional>
class OffScreenVolumeRenderer{
  public:
    struct RenderConfig{
        int fps;
        std::string backend;//cpu or cuda
        int iGPU;
        int width;
        int height;
        std::string output_video_name;
        bool save_image;
        std::string volume_data_config_file;
        float space_x,space_y,space_z;
        double lod_policy[10];
        vs::TransferFunc tf;
        std::string camera_sequence_config;
        std::string image_save_path;
    };
  public:

    OffScreenVolumeRenderer()=delete;

    /**
     * @brief read render config from json file
     * @param config_file json config file path
     */
    static void RenderFrames(const char* config_file);

    /**
     * @brief render with RenderConfig param
     */
    using Callback = std::function<void(int frame,float per,const uint8_t* data)>;
    static void RenderFrames(RenderConfig config,const Callback& callback=nullptr);

    static RenderConfig LoadRenderConfigFromFile(const char* config_file);

    static auto LoadCameraSequenceFromFile(const char* camera_file)->std::vector<vs::Camera>;
};