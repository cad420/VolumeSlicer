//
// Created by csh on 10/21/2021.
//

#pragma once

#include <QtWidgets>



#include "VolumeSlicer/volume.hpp"
#include <VolumeSlicer/transfer_function.hpp>


class TF1DEditor;
class TrivalVolume;
class VolumeRenderWidget;


using namespace vs;

class RealTimeRenderSettingWidget :public QWidget{
    Q_OBJECT
public:
    explicit RealTimeRenderSettingWidget(QWidget* parent = nullptr);
    auto getTransferFunc()->TransferFunc;
  Q_SIGNALS:
    void StartingRecord();
    void StoppedRecord();
    void updateTransferFunc(float* data,int dim);
    void updateSteps(int);
public Q_SLOTS:
    void receiveRecordStarted();
    void volumeLoaded(const std::shared_ptr<CompVolume>&);
    void volumeClosed();
    void startRecord();
    void stopRecord();
    void updateRecordPB(bool recording);
    void resetTransferFunc();
    void resetSteps();
private:
    QLabel* volume_range_label;
    QLabel* camera_pos_label;
    QDoubleSpinBox* camera_pos_x,*camera_pos_y,*camera_pos_z;

    TF1DEditor* tf_editor_widget;
    std::unique_ptr<TrivalVolume> trival_volume;

    QPushButton* start_record_pb;
    QPushButton* stop_record_pb;
    bool recording = false;

    QSlider* steps_slider;
    QSpinBox* steps_sb;
};


