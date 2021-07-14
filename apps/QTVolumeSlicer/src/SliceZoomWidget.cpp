//
// Created by wyz on 2021/6/28.
//
#include "SliceZoomWidget.hpp"
#include <QPaintEvent>
#include <QPainter>
#include <VolumeSlicer/frame.hpp>
#include <iostream>
#include <QImage>
#include <glm/glm.hpp>
SliceZoomWidget::SliceZoomWidget(QWidget *parent) {
    initSlicer();
}

void SliceZoomWidget::paintEvent(QPaintEvent *event) {
    QPainter p(this);
    Frame frame;
    frame.width=max_zoom_slicer->GetImageW();
    frame.height=max_zoom_slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
    raw_volume_sampler->Sample(max_zoom_slicer->GetSlice(),frame.data.data());

    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format::Format_Grayscale8,nullptr,nullptr);
    QImage color_img=image.convertToFormat(QImage::Format_RGBA8888);
    drawSliceLine(color_img);
    p.drawPixmap(0,0,QPixmap::fromImage(color_img.mirrored(false,true)));

    event->accept();
}

void SliceZoomWidget::mouseMoveEvent(QMouseEvent *event) {
    event->accept();
}

void SliceZoomWidget::wheelEvent(QWheelEvent *event) {
    event->accept();
}

void SliceZoomWidget::mousePressEvent(QMouseEvent *event) {
    event->accept();
}

void SliceZoomWidget::mouseReleaseEvent(QMouseEvent *event) {
    event->accept();
}

void SliceZoomWidget::setSlicer(const std::shared_ptr<Slicer> &slicer) {
    this->slicer=slicer;
    auto slice=this->slicer->GetSlice();
    slice.origin={slice.origin[0]/64,slice.origin[1]/64,slice.origin[2]/64};
    std::cout<<slice.origin[0]<<" "<<slice.origin[1]<<" "<< slice.origin[2]<<std::endl;
    slice.voxel_per_pixel_width=1;
    slice.voxel_per_pixel_height=1;
    slice.n_pixels_height=400;
    slice.n_pixels_width=400;
    this->max_zoom_slicer=Slicer::CreateSlicer(slice);
}
void SliceZoomWidget::drawSliceLine( QImage& image) {
    auto slice=slicer->GetSlice();
    assert(slice.voxel_per_pixel_height==slice.voxel_per_pixel_width);
    float p=slice.voxel_per_pixel_height/64.f;
    std::cout<<image.width()<<" "<<image.height()<<std::endl;
    auto max_zoom_slice=max_zoom_slicer->GetSlice();
//    glm::vec3 right={slice.right[0],slice.right[1],slice.right[2]};
//    glm::vec3 up={slice.up[0],slice.up[1],slice.up[2]};
//    float x_t=std::abs(glm::dot(right,{1,1,3}));
//    float y_t=std::abs(glm::dot(up,{1,1,3}));
    uint32_t min_p_x=max_zoom_slice.n_pixels_width/2-slice.n_pixels_width/2*p;
    uint32_t min_p_y=max_zoom_slice.n_pixels_height/2-slice.n_pixels_height/2*p;
    uint32_t max_p_x=max_zoom_slice.n_pixels_width/2+slice.n_pixels_width/2*p;
    uint32_t max_p_y=max_zoom_slice.n_pixels_height/2+slice.n_pixels_height/2*p;
    if(min_p_x>=0 && max_p_x<max_zoom_slice.n_pixels_width)
        for(auto i=min_p_x;i<=max_p_x;i++){
            image.setPixelColor(i,min_p_y,QColor(255,0,0,255));
            image.setPixelColor(i,max_p_y,QColor(255,0,0,255));
        }
    if(min_p_y>=0 && max_p_y<max_zoom_slice.n_pixels_height)
        for(auto i=min_p_y;i<=max_p_y;i++){
            image.setPixelColor(min_p_x,i,QColor(255,0,0,255));
            image.setPixelColor(max_p_x,i,QColor(255,0,0,255));
        }
}
void SliceZoomWidget::redraw() {
    setSlicer(this->slicer);
    repaint();
}

void SliceZoomWidget::initSlicer() {


}

void SliceZoomWidget::setRawVolume(const std::shared_ptr<RawVolume>& raw_volume) {
    this->raw_volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);
}


