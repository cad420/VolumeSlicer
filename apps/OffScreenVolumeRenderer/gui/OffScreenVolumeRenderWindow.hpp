//
// Created by wyz on 2021/11/15.
//

#pragma once

//qt
#include <QMainWindow>

#include <memory>

#include <VolumeSlicer/render.hpp>
#include "OffScreenVolumeRenderer.hpp"
using namespace vs;

class RenderFrameWidget;
class CameraVisWidget;
class RenderProgressWidget;

class OffScreenVolumeRenderWindow:public QMainWindow{
  public:
    OffScreenVolumeRenderWindow();

    explicit OffScreenVolumeRenderWindow(const std::string& config_file);

    void open(const std::string& config_file);

  private:

    /**
     * @brief this would only call in constructor OffScreenVolumeRenderWindow(const std::string& config_file)
     */
    void startRender();

    void createWidgets();

    void createMenu();

    void loadCameras(const std::string& camera_file);

  private:

    RenderFrameWidget* render_frame_widget;
    CameraVisWidget* camera_vis_widget;
    RenderProgressWidget* render_progress_widget;
    QDockWidget* render_frame_dock_widget;
    QDockWidget* camera_vis_dock_widget;
    QDockWidget* render_progress_dock_widget;

    OffScreenVolumeRenderer::RenderConfig render_config;
};