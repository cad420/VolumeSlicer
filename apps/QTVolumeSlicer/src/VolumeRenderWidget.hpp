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

namespace control{
    class TrackBallCamera;
}



/**
 * only raw volume render
 */
class VolumeRenderWidget: public QWidget{
    Q_OBJECT
public:
    explicit VolumeRenderWidget(QWidget* parent= nullptr);
    void setSlicer(const std::shared_ptr<Slicer>&);
    auto getRawVolume()->const std::shared_ptr<RawVolume>&;
    void resetTransferFunc1D(float* data,int dim=256);
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
public Q_SLOTS:
    void redraw();
    void setVisible(bool volume,bool slice);
private:
    void initTest();
private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<Slicer> dummy_slicer;
    std::shared_ptr<RawVolume> raw_volume;
    //!can render slice and volume mixed
    std::unique_ptr<RawVolumeRenderer> multi_volume_renderer;
    std::unique_ptr<control::TrackBallCamera> trackball_camera;
    std::unique_ptr<vs::Camera> base_camera;


};




#endif //VOLUMESLICER_VOLUMERENDERWIDGET_HPP
