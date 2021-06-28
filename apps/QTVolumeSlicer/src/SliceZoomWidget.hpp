//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICEZOOMWIDGET_HPP
#define VOLUMESLICER_SLICEZOOMWIDGET_HPP

#include<QtWidgets/QWidget>


class SliceZoomWidget: public QWidget{
public:
    explicit SliceZoomWidget(QWidget* parent= nullptr);
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:

};

#endif //VOLUMESLICER_SLICEZOOMWIDGET_HPP
