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
#include <VolumeSlicer/Common/frame.hpp>
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Render/camera.hpp>
#include <VolumeSlicer/Render/render.hpp>
#include <VolumeSlicer/Render/transfer_function.hpp>
#include <VolumeSlicer/Render/volume_sampler.hpp>
#include <VolumeSlicer/Utils/utils.hpp>

#include "OffScreenRenderSettingWidget.hpp"
#include "RealTimeRenderSettingWidget.hpp"
#include "RealTimeVolumeRenderWidget.hpp"

class OffScreenRenderSettingWidget;
class RealTimeRenderSettingWidget;
class RealTimeVolumeRenderWidget;
class CameraRouteWidget;

class QTOffScreenRenderEditor: public QMainWindow{
    Q_OBJECT
  public:
    explicit QTOffScreenRenderEditor(QWidget* parent = nullptr);

    void open(std::string filename);
    void close();


  private:
    void createMenu();
    void createAction();
    void createWidgets();

  private:
    QMenu* file_menu;
    QMenu* view_menu;

    QAction* open_file_action;
    QAction* close_file_action;


    RealTimeRenderSettingWidget *  setting_widget;
    QDockWidget* setting_dock_widget;

    RealTimeVolumeRenderWidget* volume_render_widget;
    QDockWidget* volume_render_dock_widget;

    OffScreenRenderSettingWidget* off_setting_widget;
    QDockWidget* off_setting_dock_widget;

};

#endif // QTOffScreenRenderEditor_QTOFFSCREENRENDEREDITOR_H
