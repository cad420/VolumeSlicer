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

    void volumeLoaded(std::string raw_volume_path,uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float space_x,float space_y, float space_z);

    void volumeClosed();

//    void getSpace(float* space);

    void getTransferFunc(float* tfData);

//    void getRenderPolicy(float* rp);

    void setTF(float* tfData, int num);

signals:

private:
    VolumeRenderWidget* volumeRenderWidget;

//    QDoubleSpinBox* space_x;
//    QDoubleSpinBox* space_y;
//    QDoubleSpinBox* space_z;

    QPushButton* startButton;
    QPushButton* stopButton;

    QSpinBox* fpsSpinBox;

    TF1DEditor* tf_editor;
    std::vector<float> tf;
    //std::vector<int> index;
    std::unique_ptr<TrivalVolume> trivalVolume;
    std::shared_ptr<RawVolume> rawVolume;

//    RenderPolicyEditor* render_policy_editor;
    std::vector<float> renderPolicy;
};

#endif // QTOFFSCREENRENDEREDITOR_SETTINGWIDGET_H
