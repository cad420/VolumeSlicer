//
// Created by wyz on 2021/11/15.
//

#pragma once

#include <QWidget>
#include <OffScreenVolumeRenderer.hpp>
QT_BEGIN_NAMESPACE
class QAction;
class QLabel;
class QMenu;

QT_END_NAMESPACE

class RenderFrameWidget: public QWidget{
  public:
    explicit RenderFrameWidget(QWidget* parent = nullptr);

  public Q_SLOTS:
    void UpdateRenderFrame(int w,int h,const uint8_t*);

  private:
    QImage image;
    QLabel* image_label;
    OffScreenVolumeRenderer::RenderConfig render_config;

};