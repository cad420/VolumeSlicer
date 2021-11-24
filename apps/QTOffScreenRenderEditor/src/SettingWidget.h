//
// Created by csh on 10/21/2021.
//

#ifndef QTOFFSCREENRENDEREDITOR_SETTINGWIDGET_H
#define QTOFFSCREENRENDEREDITOR_SETTINGWIDGET_H

#include<QtWidgets>
#include "tf1deditor.h"
#include "RenderPolicyEditor.h"
#include "VolumeRenderWidget.h"
#include "TrivalVolume.hpp"
#include "VolumeSlicer/volume.hpp"
#include "CameraRouteWidget.h"
#include "BSplineCurve.h"

class TF1DEditor;
class TrivalVolume;
class VolumeRenderWidget;
class BSplineCurve;

using namespace vs;

class SettingWidget:public QWidget{
    Q_OBJECT
public:
    explicit SettingWidget(VolumeRenderWidget* volumeRenderWidget,QWidget* parent = nullptr);

    void volumeLoaded(std::string filename);
    void volumeClosed();

    void getTransferFunc(float* tfData);

    std::string saveTFFile(std::string in_name);

    void setCameraName(std::string filename);
private:
    void initQualitySetting();
    void loadCameraSequenceFile();
    void saveSettingFile();
    void loadSettingFile();
    void optimizeCameraRoute();

signals:

private:
    VolumeRenderWidget* volumeRenderWidget;

    std::string comp_config_path;
    std::string setting_file;
    std::string volume_file;
    std::string setting_path;   //setting file's path

    QPushButton* startButton;   //start recording
    QPushButton* stopButton;    //stop recording
    QPushButton* loadCameraButton;
    QPushButton* optimizeCameraButton;
    QPushButton* loadSettingButton;
    QPushButton* saveSettingButton;

    std::string camera_sequence_config;
    std::vector<std::unique_ptr<Camera> > cameraSequence;
    std::vector<std::unique_ptr<Camera> > optimizedCameraSequence;
    int fps;

    QSlider* qualitySlider;
    std::vector<std::vector<float> > renderPolicy;

    QSpinBox* fpsSpinBox;

    TF1DEditor* tf_editor;
    std::vector<float> tf;
    std::string tfFile;

    std::unique_ptr<TrivalVolume> trivalVolume;
    std::shared_ptr<RawVolume> rawVolume;

    QCheckBox* imageSaving;
    QLineEdit* name;
};

#endif // QTOFFSCREENRENDEREDITOR_SETTINGWIDGET_H
