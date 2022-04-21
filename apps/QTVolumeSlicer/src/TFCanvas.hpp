//
// Created by wyz on 2021/7/7.
//

#pragma once
#include <QWidget>
#include <VolumeSlicer/Render/transfer_function.hpp>
using namespace vs;
class TFCanvas: public QWidget{
    Q_OBJECT
public:
    explicit TFCanvas(QWidget* parent= nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
private:

};

