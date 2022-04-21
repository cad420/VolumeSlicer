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

#include "OffScreenRenderSettingWidget.hpp"

#include "RealTimeRenderSettingWidget.hpp"
#include "camera.hpp"
#include "direct.h"
#include "json.hpp"

#include "VolumeSlicer/utils/timer.hpp"
#include <VolumeSlicer/Render/render.hpp>

class RealTimeRenderSettingWidget;
class OffScreenRenderSettingWidget;

using namespace vs;

class RealTimeVolumeRenderWidget:public QWidget{
    Q_OBJECT
public:

    explicit RealTimeVolumeRenderWidget(QWidget* parent = nullptr);

    void loadVolume(const std::string&);
    void closeVolume();
  Q_SIGNALS:
    void volumeLoaded(std::shared_ptr<CompVolume>);
    void volumeClosed();
    void recordingFinish(std::vector<Camera>);
    void recordingStart();
  public Q_SLOTS:
    void startRecording();
    void stopRecording();
    void updateTransferFunc(float* data,int dim =256);
    void updateSteps(int);
  private:
    void updateCamera();
    void recordCurrentCamera(const Camera&);
    void clearRecordCameras();
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    std::shared_ptr<CompVolume> comp_volume;
    bool recording_camera = false;

    bool left_mouse_press;
    std::unique_ptr<control::FPSCamera> control_camera;
    std::vector<Camera> recording_cameras;

    std::unique_ptr<ICompVolumeRenderer> real_time_renderer;
};

#endif // QTOffScreenRenderEditor_VOLUMERENDERWIDGET_H
