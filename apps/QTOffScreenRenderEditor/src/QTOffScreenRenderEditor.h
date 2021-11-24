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

class OffScreenRenderSettingWidget;
class SettingWidget;
class VolumeRenderWidget;
class CameraRouteWidget;

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

    SettingWidget*  setting_widget;
    QDockWidget* setting_dock_widget;

    VolumeRenderWidget* volume_render_widget;
    QDockWidget* volume_render_dock_widget;

};

#endif // QTOffScreenRenderEditor_QTOFFSCREENRENDEREDITOR_H
