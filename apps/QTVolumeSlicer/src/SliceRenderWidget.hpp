//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICERENDERWIDGET_HPP
#define VOLUMESLICER_SLICERENDERWIDGET_HPP

#include <memory>

#include<QtWidgets/QWidget>

#include <VolumeSlicer/slice.hpp>
#include <VolumeSlicer/volume_sampler.hpp>
#include <VolumeSlicer/volume.hpp>
using namespace vs;
class SliceRenderWidget;
class SliceZoomWidget;
class SliceSettingWidget;

class SliceRenderWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceRenderWidget(QWidget* parent= nullptr);
    std::shared_ptr<Slicer> getSlicer();

public:
    void initTest();

Q_SIGNALS:
    void sliceModified();

public :
    void redraw();
    void resetColorTable(float*,int dim=256);
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<CompVolume> volume;
    std::unique_ptr<VolumeSampler> volume_sampler;
    QImage color_image;
    std::vector<float> color_table;

    bool left_mouse_button_pressed;
    bool right_mouse_button_pressed;
    QPoint last_pos;

};

#endif //VOLUMESLICER_SLICERENDERWIDGET_HPP
