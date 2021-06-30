//
// Created by wyz on 2021/6/15.
//

#ifndef VOLUMESLICER_VOLUMERENDERWIDGET_HPP
#define VOLUMESLICER_VOLUMERENDERWIDGET_HPP
#include<QtWidgets/QWidget>

#include <VolumeSlicer/volume.hpp>
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/slice.hpp>

using namespace vs;

/**
 * only raw volume render
 */
class VolumeRenderWidget: public QWidget{
    Q_OBJECT
public:
    explicit VolumeRenderWidget(QWidget* parent= nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<RawVolume> raw_volume;
    //!can render slice and volume mixed
    std::unique_ptr<RawVolumeRenderer> multi_volume_renderer;
};




#endif //VOLUMESLICER_VOLUMERENDERWIDGET_HPP
