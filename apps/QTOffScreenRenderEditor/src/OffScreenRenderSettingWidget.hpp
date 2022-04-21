//
// Created by csh on 10/20/2021.
//

#pragma once

#include<QtWidgets>

#include <VolumeSlicer/Render/render.hpp>

#include "RealTimeRenderSettingWidget.hpp"

#include "RealTimeVolumeRenderWidget.hpp"
#include "VolumeSlicer/Data/volume.hpp"
#include <unordered_map>
class RenderPolicyEditor;
class TrivalVolume;
class VolumeRenderWidget;
class RealTimeRenderSettingWidget;
class CameraVisWidget;

using namespace vs;

class OffScreenRenderSettingWidget:public QWidget{
  Q_OBJECT
  public:
    explicit OffScreenRenderSettingWidget(QWidget* parent = nullptr);

    using CameraPoint = decltype(vs::Camera::pos);
    using Handle =  std::function<TransferFunc()>;
    void setTransferFuncHandle(const Handle&);
    void setLoadedVolumeFile(const std::string&);
  Q_SIGNALS:
    void volumeShouldClose();
  public Q_SLOTS:
    void receiveRecordCameras(std::vector<Camera>);
    void volumeLoaded(const std::shared_ptr<CompVolume>&);
    void volumeClosed();
  private:
    void sendCameraPosToVis(const std::vector<Camera>&);
    void importCamerasFromFile(const std::string&);
    void exportCamerasToFile(const std::string&,const std::string&);
    void deleteCamerasItem(const std::string&);
    void smoothCamerasItem(const std::string&);
    void smoothCamerasItem_v2(const std::string&);
    void clear();
    bool saveOffScreenRenderSettingToFile(const std::string&,bool);
    //todo cause too much gpu memory, so when start new render program should close the volume
    void startRenderProgram();
    auto getCurrentRenderPolicy()->std::array<double,10>;
  private:
    QListWidget* camera_item_widget;
    CameraVisWidget* camera_vis_widget;

    QPushButton* camera_load_pb;
    QPushButton* camera_export_pb;
    QPushButton* camera_del_pb;
    QSpinBox* camera_point_num_sb;
    QPushButton* smooth_camera_pb;

    std::unordered_map<std::string,std::vector<Camera>> camera_map;
    int count = 0;

    QPushButton* output_video_name_pb;
    QLineEdit* output_video_name_le;

    QSpinBox* render_width_sb;
    QSpinBox* render_height_sb;
    QSpinBox* render_fps_sb;
    QSlider* render_policy_slider;
    QComboBox* render_cameras_cb;

    QPushButton* save_render_config_pb;
    QPushButton* start_off_render_pb;

    std::array<float,10> low_render_policy;
    Handle tf_handle;
    std::string comp_volume_config;
};


