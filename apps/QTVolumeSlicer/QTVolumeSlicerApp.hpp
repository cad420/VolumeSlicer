//
// Created by wyz on 2021/6/11.
//

#ifndef VOLUMESLICER_QTVOLUMESLICERAPP_HPP
#define VOLUMESLICER_QTVOLUMESLICERAPP_HPP
#include <QMainWindow>
#include <memory>

#include<VolumeSlicer/volume.hpp>
#include<VolumeSlicer/render.hpp>
#include<VolumeSlicer/transfer_function.hpp>
#include<VolumeSlicer/camera.hpp>
#include<VolumeSlicer/frame.hpp>

using namespace vs;

class VolumeRenderWidget;

class VolumeSlicerMainWindow: public QMainWindow{
    Q_OBJECT
public:
    explicit VolumeSlicerMainWindow(QWidget* parent= nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
private:

    void createActions();
    void createMenu();

private:
    //todo: test
    void initTest();
private:
    //menu
    QMenu* m_file_menu;
    QMenu* m_file_open_menu;
    QMenu* m_view_menu;

    QAction* m_open_action;

    //widget
    VolumeRenderWidget* m_volume_render_widget;
    QDockWidget* m_volume_render_dock_widget;

    //volume and render test
    std::unique_ptr<RawVolume> raw_volume;
    std::unique_ptr<RawVolumeRenderer> multi_renderer;
    std::shared_ptr<Slicer> slicer;
};




#endif //VOLUMESLICER_QTVOLUMESLICERAPP_HPP
