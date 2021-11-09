//
// Created by csh on 10/20/2021.
//

#ifndef QTOffScreenRenderEditor_OFFSCREENRENDERSETTING_H
#define QTOffScreenRenderEditor_OFFSCREENRENDERSETTING_H

#include<QtWidgets>

#include <VolumeSlicer/render.hpp>

#include "RenderPolicyEditor.h"
#include "VolumeRenderWidget.h"
#include "SettingWidget.h"
#include "VolumeSlicer/volume.hpp"

class RenderPolicyEditor;
class TrivalVolume;
class VolumeRenderWidget;
class SettingWidget;

using namespace vs;

class OffScreenRenderSettingWidget:public QWidget{
  Q_OBJECT
  public:
    OffScreenRenderSettingWidget(VolumeRenderWidget* volumeRenderWidget, SettingWidget* settingWidget, QWidget* parent = nullptr);

    void volumeLoaded(std::string comp_volume_config);

    void volumeClosed();

  signals:

  private:
    void initCameraDialog();
  private:
    VolumeRenderWidget* volumeRenderWidget;
    SettingWidget* settingWidget;

    std::unique_ptr<CUDAOffScreenCompVolumeRenderer> offScreenRenderer;
    std::shared_ptr<CompVolume> volumeForOffScreen;

    QSlider* qualitySlider;

//    QDoubleSpinBox* space_x;
//    QDoubleSpinBox* space_y;
//    QDoubleSpinBox* space_z;

    QPushButton* loadCameraButton;
    QPushButton* editCameraButton;
    QLabel* cameraFileLabel;
    QDialog* cameraDialog;
    QListWidget* cameraList;
    QPushButton* quitButton;
    QDoubleSpinBox* zoomBox;
    QDoubleSpinBox* posxBox;
    QDoubleSpinBox* posyBox;
    QDoubleSpinBox* poszBox;
    QDoubleSpinBox* lookAtxBox;
    QDoubleSpinBox* lookAtyBox;
    QDoubleSpinBox* lookAtzBox;
    QDoubleSpinBox* upxBox;
    QDoubleSpinBox* upyBox;
    QDoubleSpinBox* upzBox;
//    QDoubleSpinBox* rightxBox;
//    QDoubleSpinBox* rightyBox;
//    QDoubleSpinBox* rightzBox;

    QPushButton* startButton;
    QPushButton* saveButton;

    QCheckBox* imageSaving;
    QLineEdit* name;

    int frameHeight;
    int frameWidth;
    std::string output_video_name;
    std::string volume_data_config;
    std::vector<float> space;
    std::vector<float> render_policy;
    std::vector<float> tf;
    std::string camera_sequence_config;
    std::vector<std::unique_ptr<Camera> > cameraSequence;

//    "fps": 30,
//    "backend": "cuda",
//    "iGPU": 0,
//    "width": 900,
//    "height": 900,
//    "output_video_name": "result33.avi",
//    "save_image": "yes",
//    "volume_data_config": "C:/Users/csh/project/VolumeSlicer/bin/mouse_file_config.json",
//    "space": [0.00032,0.00032,0.001],
//    "lod_policy": [0.8,1.6,3.2,6.4,8.0,9.6,-1.0],
//    "tf": {
//        "0": [0.0,0.0,0.0,0.0],
//        "74": [0.0,0.0,0.0,0.0],
//        "75": [0.75,0.75,0.75,0.6],
//        "255": [1.0,1.0,1.0,1.0]
//    },
//    "camera_sequence_config": "camera_sequence_config.json"
};

#endif // QTOffScreenRenderEditor_OFFSCREENRENDERSETTING_H