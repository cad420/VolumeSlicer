//
// Created by wyz on 2021/6/15.
//
#pragma once
#include<QtWidgets/QWidget>

#include <VolumeSlicer/Data/slice.hpp>
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Render/render.hpp>
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
    void loadVolume(const char*,const std::array<uint32_t,3>&,const std::array<float,3>&);
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
    void volumeLoaded();
    void volumeClose();
private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<Slicer> dummy_slicer;
    std::shared_ptr<RawVolume> raw_volume;
    //!can render slice and volume mixed
    std::unique_ptr<SliceRawVolumeMixRenderer> multi_volume_renderer;
    std::unique_ptr<control::TrackBallCamera> trackball_camera;
    std::unique_ptr<vs::Camera> base_camera;


};


