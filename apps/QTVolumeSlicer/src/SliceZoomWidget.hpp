//
// Created by wyz on 2021/6/28.
//

#pragma once

#include<QtWidgets/QWidget>

#include <VolumeSlicer/Data/slice.hpp>
#include <VolumeSlicer/Render/volume_sampler.hpp>
using namespace vs;

class SliceZoomWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceZoomWidget(QWidget* parent= nullptr);
    void setSlicer(const std::shared_ptr<Slicer>& slicer);
    void setRawVolume(const std::shared_ptr<RawVolume>& raw_volume);
private:
    void initSlicer();
    void drawSliceLine( QImage& image);
public Q_SLOTS:
    void redraw();
    void volumeLoaded();
    void volumeClose();
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<Slicer> max_zoom_slicer;
    std::unique_ptr<VolumeSampler> raw_volume_sampler;
    std::shared_ptr<RawVolume> raw_volume;

};


