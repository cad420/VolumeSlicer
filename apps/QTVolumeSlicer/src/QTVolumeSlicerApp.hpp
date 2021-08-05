//
// Created by wyz on 2021/6/11.
//

#ifndef VOLUMESLICER_QTVOLUMESLICERAPP_HPP
#define VOLUMESLICER_QTVOLUMESLICERAPP_HPP
//qt
#include <QMainWindow>
//std
#include <memory>
//vs
#include<VolumeSlicer/volume.hpp>
#include<VolumeSlicer/render.hpp>
#include<VolumeSlicer/transfer_function.hpp>
#include<VolumeSlicer/camera.hpp>
#include<VolumeSlicer/frame.hpp>
#include<VolumeSlicer/volume_sampler.hpp>

using namespace vs;

class VolumeRenderWidget;
class SliceRenderWidget;
class SliceZoomWidget;
class SliceSettingWidget;
class VolumeSettingWidget;
class VolumeRenderSettingWidget;

//qt
class QScrollArea;
class QComboBox;
class QToolButton;
class QButtonGroup;

class VolumeSlicerMainWindow: public QMainWindow{
    Q_OBJECT
public:
    explicit VolumeSlicerMainWindow(QWidget* parent= nullptr);

public:
    void open(const std::string& file_name);

private:

    void createActions();
    void createMenu();
    void createWidgets();
    void createToolBar();
    void setModulePanel(QWidget* widget);

private:
    //menu
    QMenu* m_file_menu;

    QMenu* m_view_menu;

    //actions
    QAction* m_open_action;
    QAction* m_volume_setting_action;
    QAction* m_slice_setting_action;
    QAction* m_volume_render_setting_action;
    QAction* m_slice_default_zoom_action;
    QAction* m_capture_slice_action;
    QAction* m_capture_volume_action;

    //tool bar
    QToolBar* m_tool_bar;
    QComboBox* m_module_panel;

    //tool button
    QToolButton* m_slice_zoom_in_tool_button;
    QToolButton* m_slice_zoom_out_tool_button;

    //status bar
    QStatusBar* m_status_bar;



    //widget
    VolumeRenderWidget* m_volume_render_widget;
    QDockWidget* m_volume_render_dock_widget;

    SliceRenderWidget* m_slice_render_widget;
    QDockWidget* m_slice_render_dock_widget;

    SliceZoomWidget* m_slice_zoom_widget;
    QDockWidget* m_slice_zoom_dock_widget;

    SliceSettingWidget* m_slice_setting_widget;
    VolumeSettingWidget* m_volume_setting_widget;
    VolumeRenderSettingWidget* m_volume_render_setting_widget;



    QDockWidget* m_setting_dock_widget;



    //volume and render test
    std::shared_ptr<RawVolume> raw_volume;
    std::unique_ptr<RawVolumeRenderer> multi_renderer;
    std::shared_ptr<Slicer> slicer;
    std::unique_ptr<VolumeSampler> volume_sampler;
    std::shared_ptr<CompVolume> comp_volume;
    std::unique_ptr<VolumeSampler> comp_volume_sampler;

    bool left_mouse_button_pressed;
    QPoint last_pos;
};




#endif //VOLUMESLICER_QTVOLUMESLICERAPP_HPP
