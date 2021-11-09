//
// Created by csh on 10/20/2021.
//

#ifndef QTOffScreenRenderEditor_QTOFFSCREENRENDEREDITOR_H
#define QTOffScreenRenderEditor_QTOFFSCREENRENDEREDITOR_H
//qt
#include <QMainWindow>
#include <QtWidgets>
//std
#include <memory>
//vs
#include<VolumeSlicer/volume.hpp>
#include<VolumeSlicer/render.hpp>
#include<VolumeSlicer/transfer_function.hpp>
#include<VolumeSlicer/camera.hpp>
#include<VolumeSlicer/frame.hpp>
#include<VolumeSlicer/volume_sampler.hpp>
#include <VolumeSlicer/utils.hpp>

#include "SettingWidget.h"
#include "VolumeRenderWidget.h"
#include "OffScreenRenderSetting.h"

//class RealTimeRenderSettingWidget;
class OffScreenRenderSettingWidget;
//class VolumeRenderWidget;
class SettingWidget;
class VolumeRenderWidget;

class QTOffScreenRenderEditor: public QMainWindow{
    Q_OBJECT
  public:
    explicit QTOffScreenRenderEditor(QWidget* parent = nullptr);

    void open(std::string filename);

  private:
    void CreateMenu();
    void CreateAction();
    void CreateWidgets();

  private:
    QMenu* file_menu;
    QMenu* view_menu;

    QAction* open_file_action;
//    QAction* real_time_setting_action;
//    QAction* off_screen_setting_action;
    QAction* volume_render_action;


    SettingWidget*  setting_widget;
    QDockWidget* setting_dock_widget;

    VolumeRenderWidget* volume_render_widget;
    QDockWidget* volume_render_dock_widget;
//
//    RealTimeRenderSettingWidget* realtime_render_setting_widget;
//    QDockWidget* realtime_render_setting_dock_widget;
//
    OffScreenRenderSettingWidget* offscreen_render_setting_widget;
    QDockWidget* offscreen_render_setting_dock_widget;


};

#endif // QTOffScreenRenderEditor_QTOFFSCREENRENDEREDITOR_H
