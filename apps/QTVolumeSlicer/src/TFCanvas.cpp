//
// Created by wyz on 2021/7/7.
//
#include "TFCanvas.hpp"
#include <QStyleOption>
#include <QPainter>
TFCanvas::TFCanvas(QWidget *parent) {

}
void TFCanvas::paintEvent(QPaintEvent *event) {
    QStyleOption opt;
    opt.initFrom(this);
    QPainter p(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
    QWidget::paintEvent(event);
}
void TFCanvas::mouseMoveEvent(QMouseEvent *event) {
    QWidget::mouseMoveEvent(event);
}
void TFCanvas::mousePressEvent(QMouseEvent *event) {
    QWidget::mousePressEvent(event);
}

void TFCanvas::mouseReleaseEvent(QMouseEvent *event) {
    QWidget::mouseReleaseEvent(event);
}

void TFCanvas::mouseDoubleClickEvent(QMouseEvent *event) {
    QWidget::mouseDoubleClickEvent(event);
}

void TFCanvas::keyPressEvent(QKeyEvent *event) {
    QWidget::keyPressEvent(event);
}

void TFCanvas::keyReleaseEvent(QKeyEvent *event) {
    QWidget::keyReleaseEvent(event);
}