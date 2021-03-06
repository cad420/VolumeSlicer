//
// Created by wyz on 2021/6/28.
//

#pragma once

#include <memory>

#include<QtWidgets/QWidget>

#include <VolumeSlicer/Data/slice.hpp>
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Render/volume_sampler.hpp>
using namespace vs;
class SliceRenderWidget;
class SliceZoomWidget;
class SliceSettingWidget;

class SliceRenderWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceRenderWidget(QWidget* parent= nullptr);
    std::shared_ptr<Slicer> getSlicer();

public:
    bool loadVolume(const char* file_path,const std::array<float,3>&);
    auto getCompVolume()->std::shared_ptr<CompVolume>;
Q_SIGNALS:
    void sliceModified();
public Q_SLOTS:
    void volumeLoaded();
    void volumeClose();
public :
    void redraw();
    void resetColorTable(float*,int dim=256);
protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
private:
    std::shared_ptr<Slicer> slicer;
    std::shared_ptr<CompVolume> volume;
    std::unique_ptr<VolumeSampler> volume_sampler;
    QImage color_image;
    std::vector<float> color_table;

    bool left_mouse_button_pressed;
    bool right_mouse_button_pressed;
    QPoint last_pos;

};


