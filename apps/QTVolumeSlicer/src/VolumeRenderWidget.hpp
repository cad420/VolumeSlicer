//
// Created by wyz on 2021/6/15.
//

#ifndef VOLUMESLICER_VOLUMERENDERWIDGET_HPP
#define VOLUMESLICER_VOLUMERENDERWIDGET_HPP

#include<QtWidgets/QWidget>


class VolumeRenderWidget: public QWidget{
public:
    explicit VolumeRenderWidget(QWidget* parent= nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:

};




#endif //VOLUMESLICER_VOLUMERENDERWIDGET_HPP
