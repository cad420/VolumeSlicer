//
// Created by wyz on 2021/6/28.
//
#pragma once

#include<QtWidgets/QWidget>
class VolumeRenderWidget;
class TransferFunctionWidget;
class TF1DEditor;
class TrivalVolume;

class VolumeRenderSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit VolumeRenderSettingWidget(VolumeRenderWidget* widget,QWidget* parent= nullptr);

public Q_SLOTS:
    void volumeLoaded();
    void volumeClose();
private:
    VolumeRenderWidget* m_volume_render_widget;
    TransferFunctionWidget* tf_widget;
    TF1DEditor* tf_editor_widget;
    std::unique_ptr<TrivalVolume> trival_volume;
    std::vector<float> tf;
};

