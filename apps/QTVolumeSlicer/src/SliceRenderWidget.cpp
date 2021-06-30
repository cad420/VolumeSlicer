//
// Created by wyz on 2021/6/28.
//
#include"SliceRenderWidget.hpp"
#include <VolumeSlicer/frame.hpp>
#include <VolumeSlicer/utils.hpp>

#include <QPainter>
#include <QImage>
#include <QMouseEvent>
SliceRenderWidget::SliceRenderWidget(QWidget *parent) {
    initTest();
}

void SliceRenderWidget::paintEvent(QPaintEvent *event) {
    std::cout<<__FUNCTION__ <<std::endl;
//    QPainter p(this);
//    Frame frame;
//    frame.width=slicer->GetImageW();
//    frame.height=slicer->GetImageH();
//    frame.channels=1;
//    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
//    bool complete;
//
//    complete=volume_sampler->Sample(slicer->GetSlice(),frame.data.data());
//
//
//    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format::Format_Grayscale8,nullptr,nullptr);
//
//    p.drawPixmap(0,0,QPixmap::fromImage(image.mirrored(false,true)));

}

void SliceRenderWidget::mouseMoveEvent(QMouseEvent *event) {
    QWidget::mouseMoveEvent(event);
}

void SliceRenderWidget::wheelEvent(QWheelEvent *event) {
    QWidget::wheelEvent(event);
}

void SliceRenderWidget::mousePressEvent(QMouseEvent *event) {
    repaint();
    event->accept();

}

void SliceRenderWidget::mouseReleaseEvent(QMouseEvent *event) {
    QWidget::mouseReleaseEvent(event);
}

void SliceRenderWidget::keyPressEvent(QKeyEvent *event) {
    QWidget::keyPressEvent(event);
}

std::shared_ptr<Slicer> SliceRenderWidget::getSlicer() {
    return this->slicer;
}

void SliceRenderWidget::initTest() {
//    Slice slice;
//    slice.origin={9765.f,8434.f,13698.f,1.f};
//    slice.right={1.f,0.f,0.f,0.f};
//    slice.up={0.f,0.f,-1.f,0.f};
//    slice.normal={0.f,1.f,0.f,0.f};
//    slice.n_pixels_width=1200;
//    slice.n_pixels_height=900;
//    slice.voxel_per_pixel_height=2.f;
//    slice.voxel_per_pixel_width=2.f;
//    slicer=Slicer::CreateSlicer(slice);
//
//    volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
//    volume_sampler=VolumeSampler::CreateVolumeSampler(volume);
//    volume->SetSpaceX(0.01f);
//    volume->SetSpaceY(0.01f);
//    volume->SetSpaceZ(0.03f);
//    auto block_length=volume->GetBlockLength();
//    std::cout<<"block length: "<<block_length[0]<<" "<<block_length[1]<<std::endl;
//    auto block_dim=volume->GetBlockDim(0);
//    std::cout<<"block dim: "<<block_dim[0]<<" "<<block_dim[1]<<" "<<block_dim[2]<<std::endl;
}
