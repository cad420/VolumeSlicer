//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICEZOOMWIDGET_HPP
#define VOLUMESLICER_SLICEZOOMWIDGET_HPP

#include<QtWidgets/QWidget>

#include <VolumeSlicer/slice.hpp>
using namespace vs;

class SliceZoomWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceZoomWidget(QWidget* parent= nullptr);
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<Slicer> max_zoom_slicer;
};

#endif //VOLUMESLICER_SLICEZOOMWIDGET_HPP
