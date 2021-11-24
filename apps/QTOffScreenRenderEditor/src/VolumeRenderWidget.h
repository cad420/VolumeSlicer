//
// Created by csh on 10/20/2021.
//

#ifndef QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H
#define QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H

#include <fstream>
#include <ctime>

#include<QtWidgets>
#include <QImage>

#include "OffScreenRenderSetting.h"

#include <VolumeSlicer/render.hpp>
#include "json.hpp"
#include "camera.hpp"
#include "SettingWidget.h"
#include "VideoCapture.hpp"
#include "utils/timer.hpp"
#include "direct.h"
#include <filesystem>
#include <spdlog/sinks/rotating_file_sink.h>

class SettingWidget;
class VideoCapture;
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

    //void offScreenRender();
    void startRecording();
    void stopRecording();

    void draw();

    //int getFPS();

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
//    OffScreenRenderSettingWidget* offScreenRenderSettingWidget;

    std::unique_ptr<OpenGLCompVolumeRenderer> realTimeRenderer;
    std::shared_ptr<CompVolume> volumeForRealTime;
//    std::shared_ptr<RawVolume> rawVolume;

//    std::unique_ptr<CUDAOffScreenCompVolumeRenderer> offScreenRenderer;
//    std::shared_ptr<CompVolume> volumeForOffScreen;

    std::unique_ptr<Camera> baseCamera;
    std::unique_ptr<control::FPSCamera> fpsCamera;
    float moveSpeed;
    std::vector<std::unique_ptr<Camera> > cameraSequence;

    QTimer* timer;
    //std::string output_file_name;
    int fps;

    bool left_pressed;
    //bool save_image;

    //std::string m_comp_volume_path;

    int sequenceNum;

    std::vector<float> space;

    std::string curCameraFile;
};

#endif // QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H
