//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICERENDERWIDGET_HPP
#define VOLUMESLICER_SLICERENDERWIDGET_HPP

#include<QtWidgets/QWidget>


class SliceRenderWidget: public QWidget{
public:
    explicit SliceRenderWidget(QWidget* parent= nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:

};

#endif //VOLUMESLICER_SLICERENDERWIDGET_HPP
