//
// Created by wyz on 2021/11/15.
//
#include "RenderFrameWidget.hpp"
#include <QLabel>
#include <QVBoxLayout>
#include <QGridLayout>
#include <VolumeSlicer/Utils/logger.hpp>
RenderFrameWidget::RenderFrameWidget(QWidget *parent)
:QWidget(parent)
{
    image_label = new QLabel;
    image_label->setBackgroundRole(QPalette::Base);
    image_label->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
    image_label->setScaledContents(true);
    image_label->setFixedSize(480,270);
    image_label->setAlignment(Qt::AlignCenter);
    auto layout = new QGridLayout;
    layout->addWidget(image_label);
    setLayout(layout);

}

void RenderFrameWidget::UpdateRenderFrame(int w, int h, const uint8_t *data)
{
    assert(data);
    image = QImage(data,w,h,QImage::Format::Format_RGBA8888);

    image_label->setPixmap(QPixmap::fromImage(image));

}
