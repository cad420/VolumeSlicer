//
// Created by wyz on 2021/6/28.
//
#include"SliceRenderWidget.hpp"
#include <VolumeSlicer/frame.hpp>
#include <VolumeSlicer/utils.hpp>

#include <QPainter>
#include <QImage>
#include <QMouseEvent>
#include <QApplication>
#include <QKeyEvent>
SliceRenderWidget::SliceRenderWidget(QWidget *parent) {
    initTest();
}

void SliceRenderWidget::paintEvent(QPaintEvent *event) {
//    std::cout<<__FUNCTION__ <<std::endl;
    QPainter p(this);
    Frame frame;
    frame.width=slicer->GetImageW();
    frame.height=slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
    bool complete;

    complete=volume_sampler->Sample(slicer->GetSlice(),frame.data.data());


    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format::Format_Grayscale8,nullptr,nullptr);

    p.drawPixmap(0,0,QPixmap::fromImage(image.mirrored(false,true)));

}

void SliceRenderWidget::mouseMoveEvent(QMouseEvent *event) {
    if(left_mouse_button_pressed){
        auto pos=event->pos();
        auto d=last_pos-pos;
        last_pos=pos;
        std::cout<<d.x()<<" "<<d.y()<<std::endl;
        slicer->MoveInPlane(d.x(),d.y());
    }
    event->accept();
    repaint();
    emit sliceModified();
}

void SliceRenderWidget::wheelEvent(QWheelEvent *event) {
//    std::cout<<__FUNCTION__ <<std::endl;
    auto angle_delta=event->angleDelta();
    if((QApplication::keyboardModifiers() == Qt::ControlModifier)){
        spdlog::info("{0}",__FUNCTION__ );
        if(angle_delta.y()>0){
            slicer->StretchInXY(1.1f,1.1f);
        }
        else{
            slicer->StretchInXY(0.9f,0.9f);
        }
    }
    else{
        if(angle_delta.y()>0)
            slicer->MoveByNormal(1.f);
        else
            slicer->MoveByNormal(-1.f);
    }
    event->accept();
    repaint();
    emit sliceModified();
}

void SliceRenderWidget::mousePressEvent(QMouseEvent *event) {
//    std::cout<<__FUNCTION__ <<std::endl;
    setFocus();

    if(event->button()==Qt::MouseButton::LeftButton){
        left_mouse_button_pressed=true;
    }
    else if(event->button()==Qt::MouseButton::RightButton){
        right_mouse_button_pressed=true;
    }
    last_pos=event->pos();
    event->accept();
    repaint();
    emit sliceModified();
}

void SliceRenderWidget::mouseReleaseEvent(QMouseEvent *event) {
//    std::cout<<__FUNCTION__ <<std::endl;
    if(event->button()==Qt::MouseButton::LeftButton){
        left_mouse_button_pressed=false;
    }
    else if(event->button()==Qt::MouseButton::RightButton){
        right_mouse_button_pressed=false;
        auto cur_pos=event->pos();
        auto delta=last_pos-cur_pos;
        slicer->RotateByZ(90.0/180.0*3.141592627);
//        slicer->RotateByY(delta.y());
    }
    event->accept();
    repaint();
    emit sliceModified();
}

void SliceRenderWidget::keyPressEvent(QKeyEvent *event) {
//    std::cout<<__FUNCTION__ <<std::endl;
    switch (event->key()) {
        case 'A':slicer->RotateByX(1.0/180.0*3.141592627);
        std::cout<<"a press"<<std::endl;
        break;
        case 'D':slicer->RotateByX(-1.0/180.0*3.141592627);break;
        case 'W':slicer->RotateByY(1.0/180.0*3.141592627);break;
        case 'S':slicer->RotateByY(-1.0/180.0*3.141592627);break;
        case 'Q':slicer->RotateByZ(1.0/180.0*3.141592627);break;
        case 'E':slicer->RotateByZ(-1.0/180.0*3.141592627);break;
    }
    event->accept();
    repaint();
    emit sliceModified();
}

std::shared_ptr<Slicer> SliceRenderWidget::getSlicer() {
    return this->slicer;
}

void SliceRenderWidget::initTest() {
    Slice slice;
    slice.origin={9765.f,8434.f,4500.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,0.f,-1.f,0.f};
    slice.normal={0.f,1.f,0.f,0.f};
    slice.n_pixels_width=1200;
    slice.n_pixels_height=900;
    slice.voxel_per_pixel_height=2.f;
    slice.voxel_per_pixel_width=2.f;
    slicer=Slicer::CreateSlicer(slice);

    volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
    volume_sampler=VolumeSampler::CreateVolumeSampler(volume);
    volume->SetSpaceX(0.01f);
    volume->SetSpaceY(0.01f);
    volume->SetSpaceZ(0.03f);
    auto block_length=volume->GetBlockLength();
    std::cout<<"block length: "<<block_length[0]<<" "<<block_length[1]<<std::endl;
    auto block_dim=volume->GetBlockDim(0);
    std::cout<<"block dim: "<<block_dim[0]<<" "<<block_dim[1]<<" "<<block_dim[2]<<std::endl;
}

void SliceRenderWidget::redraw() {
    repaint();
    emit sliceModified();
}
