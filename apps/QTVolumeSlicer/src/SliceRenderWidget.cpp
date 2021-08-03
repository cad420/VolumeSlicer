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
#include <omp.h>
SliceRenderWidget::SliceRenderWidget(QWidget *parent) {

    initTest();
    color_image=QImage(slicer->GetImageW(),slicer->GetImageH(),QImage::Format_RGBA8888);
    color_table.resize(256*4);
    for(int i=0;i<256;i++){
        color_table[i*4]=color_table[i*4+1]=color_table[i*4+2]=i/255.f;
        color_table[i*4+3]=1.f;
    }
}

void SliceRenderWidget::paintEvent(QPaintEvent *event) {

    std::cout<<"slice paint event"<<std::endl;
//    std::cout<<__FUNCTION__ <<std::endl;
    START_CPU_TIMER
    QPainter p(this);
    Frame frame;
    frame.width=slicer->GetImageW();
    frame.height=slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
    bool complete;

    complete=volume_sampler->Sample(slicer->GetSlice(),frame.data.data());


//#pragma omp parallel for
    for(int i=0;i<frame.width;i++){
        for(int j=0;j<frame.height;j++){
            size_t idx=(size_t)i*frame.width+j;
            int scalar=frame.data[idx];
            color_image.setPixelColor(j,frame.width-1-i,QColor(
                    color_table[scalar*4]*255,color_table[scalar*4+1]*255,color_table[scalar*4+2]*255,255
                    ));
        }
    }
    p.drawImage(0,0,color_image);


//    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format_Grayscale8);
//    p.drawPixmap(0,0,QPixmap::fromImage(image.mirrored(false,true)));
    END_CPU_TIMER
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
        auto lod=slicer->GetSlice().voxel_per_pixel_width;
        if(angle_delta.y()>0)
            slicer->MoveByNormal(lod);
        else
            slicer->MoveByNormal(-lod);
    }
    event->accept();
    repaint();
    emit sliceModified();
}

void SliceRenderWidget::mousePressEvent(QMouseEvent *event) {
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
    std::cout<<__FUNCTION__ <<" "<<event->key()<<std::endl;
    switch (event->key()) {
        case 'A':slicer->RotateByX(1.0/180.0*3.141592627);
        std::cout<<"a press"<<std::endl;
        break;
        case 'D':slicer->RotateByX(-1.0/180.0*3.141592627);break;
        case 'W':slicer->RotateByY(1.0/180.0*3.141592627);break;
        case 'S':slicer->RotateByY(-1.0/180.0*3.141592627);break;
        case 'Q':slicer->RotateByZ(1.0/180.0*3.141592627);break;
        case 'E':slicer->RotateByZ(-1.0/180.0*3.141592627);break;
        case Qt::Key_Left:;
        case Qt::Key_Down:{
            auto lod=slicer->GetSlice().voxel_per_pixel_width;
            slicer->MoveByNormal(-lod);
            break;
        }
        case Qt::Key_Right:;
        case Qt::Key_Up:{
            auto lod=slicer->GetSlice().voxel_per_pixel_width;
            slicer->MoveByNormal(lod);
            break;
        }

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
    slice.n_pixels_width=900;
    slice.n_pixels_height=900;
    slice.voxel_per_pixel_height=2.f;
    slice.voxel_per_pixel_width=2.f;
    slicer=Slicer::CreateSlicer(slice);

    SetCUDACtx(0);
    volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
    volume_sampler=VolumeSampler::CreateVolumeSampler(volume);
    volume->SetSpaceX(0.00032f);
    volume->SetSpaceY(0.00032f);
    volume->SetSpaceZ(0.001f);
    slicer->SetSliceSpaceRatio({1,1,0.001f/0.00032f});
    auto block_length=volume->GetBlockLength();
    std::cout<<"block length: "<<block_length[0]<<" "<<block_length[1]<<std::endl;
    auto block_dim=volume->GetBlockDim(0);
    std::cout<<"block dim: "<<block_dim[0]<<" "<<block_dim[1]<<" "<<block_dim[2]<<std::endl;
}

void SliceRenderWidget::redraw() {
    repaint();
    emit sliceModified();
}

void SliceRenderWidget::resetColorTable(float *tf, int dim) {
    memcpy(color_table.data(),tf,dim*4*sizeof(float));
}
