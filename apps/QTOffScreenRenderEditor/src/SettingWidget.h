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


class TF1DEditor;
class RenderPolicyEditor;
class TrivalVolume;
class VolumeRenderWidget;

using  namespace vs;

class SettingWidget:public QWidget{
    Q_OBJECT
public:
    SettingWidget(VolumeRenderWidget* volumeRenderWidget, QWidget* parent = nullptr);

    void volumeLoaded(std::string file, std::string comp_volume_config,std::string raw_volume_path,uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float space_x,float space_y, float space_z);
    void volumeLoaded(std::string filename);
    void volumeClosed();

//    void getSpace(float* space);

    void getTransferFunc(float* tfData);

//    void getRenderPolicy(float* rp);

    void setTF(float* tfData, int num);

    std::string saveTFFile();

    void setCameraName(std::string filename);
private:
    void initQualitySetting();
    void loadCameraSequenceFile();
    void saveSettingFile();
    void loadSettingFile();

signals:

private:
    VolumeRenderWidget* volumeRenderWidget;
//
//    int frameHeight;
//    int frameWidth;

//    QDoubleSpinBox* space_x;
//    QDoubleSpinBox* space_y;
//    QDoubleSpinBox* space_z;

    QLabel* volumeFileLabel;

    std::string setting_file;
    std::string volume_file;
    std::string setting_path;

    QPushButton* startButton;
    QPushButton* stopButton;
//    QPushButton* saveTFButton;

    QPushButton* loadCameraButton;
    QPushButton* optimizeCameraButton;
    std::string camera_sequence_config;
    std::vector<std::unique_ptr<Camera> > cameraSequence;
    int fps;

    QSlider* qualitySlider;
    std::vector<std::vector<float> > renderPolicy;

    QSpinBox* fpsSpinBox;

    TF1DEditor* tf_editor;
    std::vector<float> tf;
    std::string tfFile;
    std::unique_ptr<TrivalVolume> trivalVolume;
    std::shared_ptr<RawVolume> rawVolume;

    std::string settingFile;
    QPushButton* loadSettingButton;
    QPushButton* saveSettingButton;

    QCheckBox* imageSaving;
    QLineEdit* name;

    std::string comp_config_path;

//    RenderPolicyEditor* render_policy_editor;
//    std::vector<float> renderPolicy;

};

#endif // QTOFFSCREENRENDEREDITOR_SETTINGWIDGET_H
