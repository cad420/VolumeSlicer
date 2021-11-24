//
// Created by csh on 10/20/2021.
//

#ifndef QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H
#define QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H

#include <fstream>
#include <ctime>
#include <filesystem>

#include<QtWidgets>
#include <QImage>

#include "OffScreenRenderSetting.h"

#include "json.hpp"
#include "camera.hpp"
#include "SettingWidget.h"
#include "direct.h"

#include "VolumeSlicer/utils/timer.hpp"
#include <VolumeSlicer/render.hpp>

#include <spdlog/sinks/rotating_file_sink.h>

class SettingWidget;
class OffScreenRenderSettingWidget;

using namespace vs;

class VolumeRenderWidget:public QWidget{
    Q_OBJECT
public:

    explicit VolumeRenderWidget(QWidget* parent = nullptr);

    void loadVolume(const std::string& volume_config_file_path);

    void setFPS(int in_fps);
    void setTransferFunction(const float* tfData, int num);
    void setRenderPolicy(const float* renderPolicy,int num);

    std::shared_ptr<CompVolume> getVolumeForRealTime();

    void volumeClosed();

    void setWidget(SettingWidget* in_settingWidget);

    void startRecording();
    void stopRecording();

    void draw();

private:
    void updateCamera();
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    int frameHeight;
    int frameWidth;

    SettingWidget* settingWidget;

    std::unique_ptr<OpenGLCompVolumeRenderer> realTimeRenderer;
    std::shared_ptr<CompVolume> volumeForRealTime;

    std::unique_ptr<Camera> baseCamera;
    std::unique_ptr<control::FPSCamera> fpsCamera;
    float moveSpeed;
    std::vector<std::unique_ptr<Camera> > cameraSequence;

    QTimer* timer;

    int fps;

    bool left_pressed;

    std::vector<float> space;

    std::string curCameraFile;  //camera sequence file path
};

#endif // QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H
