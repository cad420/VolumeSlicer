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

}

void SliceRenderWidget::paintEvent(QPaintEvent *event) {

//    std::cout<<"slice paint event"<<std::endl;
    if(!slicer || !volume || !volume_sampler) return;
    START_CPU_TIMER
    QPainter p(this);
    Frame frame;
    frame.width=slicer->GetImageW();
    frame.height=slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
    bool complete;

    complete=volume_sampler->Sample(slicer->GetSlice(),frame.data.data(),true);


//#pragma omp parallel for
    for(int i=0;i<frame.width;i++){
        for(int j=0;j<frame.height;j++){
            size_t idx=(size_t)j*frame.width+i;
            int scalar=frame.data[idx];
            color_image.setPixelColor(i,frame.height-1-j,QColor(
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
    if(!slicer) return;
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
    if(!slicer) return;
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
    if(!slicer) return;

    if(event->button()==Qt::MouseButton::LeftButton){
        left_mouse_button_pressed=true;
    }
    else if(event->button()==Qt::MouseButton::RightButton){
        right_mouse_button_pressed=true;
    }
    last_pos=event->pos();
    event->accept();
    repaint();
//    emit sliceModified();
}

void SliceRenderWidget::mouseReleaseEvent(QMouseEvent *event) {
    if(!slicer) return;
    if(event->button()==Qt::MouseButton::LeftButton){
        left_mouse_button_pressed=false;
    }
    else if(event->button()==Qt::MouseButton::RightButton){
        right_mouse_button_pressed=false;
//        auto cur_pos=event->pos();
//        auto delta=last_pos-cur_pos;
//        slicer->RotateByZ(90.0/180.0*3.141592627);
//        slicer->RotateByY(delta.y());
    }
    event->accept();
    repaint();
//    emit sliceModified();
}

void SliceRenderWidget::keyPressEvent(QKeyEvent *event) {
    if(!slicer) return;
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

bool SliceRenderWidget::loadVolume(const char *file_path,const std::array<float,3>& space) {
    SetCUDACtx(0);

    PrintCUDAMemInfo("before comp volume load");
    volume=CompVolume::Load(file_path);
    PrintCUDAMemInfo("after comp volume load");

    PrintCUDAMemInfo("before comp volume sampler create");
    volume_sampler=VolumeSampler::CreateVolumeSampler(volume);
    PrintCUDAMemInfo("after comp volume sampler create");
    volume->SetSpaceX(space[0]);
    volume->SetSpaceY(space[1]);
    volume->SetSpaceZ(space[2]);
    auto base_space=std::min({space[0],space[1],space[2]});


    Slice slice;
    slice.origin={volume->GetVolumeDimX()/2.f,
                  volume->GetVolumeDimY()/2.f,
                  volume->GetVolumeDimZ()/2.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,0.f,-1.f,0.f};
    slice.normal={0.f,1.f,0.f,0.f};
    slice.n_pixels_width=this->width();
    slice.n_pixels_height=this->height();
    slice.voxel_per_pixel_height=3.f;
    slice.voxel_per_pixel_width=3.f;
    slicer=Slicer::CreateSlicer(slice);
    slicer->SetSliceSpaceRatio({space[0]/base_space,
                                space[1]/base_space,
                                space[2]/base_space});


    color_image=QImage(slicer->GetImageW(),slicer->GetImageH(),QImage::Format_RGBA8888);
    color_table.resize(256*4);
    for(int i=0;i<256;i++){
        color_table[i*4]=color_table[i*4+1]=color_table[i*4+2]=i/255.f;
        color_table[i*4+3]=1.f;
    }


    return true;
}
void SliceRenderWidget::redraw() {
    if(!slicer || !volume ) return;
    repaint();
//    emit sliceModified();
}

void SliceRenderWidget::resetColorTable(float *tf, int dim) {
    memcpy(color_table.data(),tf,dim*4*sizeof(float));
}

auto SliceRenderWidget::getCompVolume() -> std::shared_ptr<CompVolume> {
    return this->volume;
}

void SliceRenderWidget::resizeEvent(QResizeEvent *event) {
    if(!volume || ! volume_sampler || !slicer) return;
    slicer->resize(event->size().width(),event->size().height());
    color_image=QImage(slicer->GetImageW(),slicer->GetImageH(),QImage::Format_RGBA8888);
    color_table.resize(256*4);
    for(int i=0;i<256;i++){
        color_table[i*4]=color_table[i*4+1]=color_table[i*4+2]=i/255.f;
        color_table[i*4+3]=1.f;
    }
    emit sliceModified();
//    QWidget::resizeEvent(event);
}

void SliceRenderWidget::volumeLoaded() {

}

void SliceRenderWidget::volumeClose() {
    spdlog::info("{0}.",__FUNCTION__ );
    slicer.reset();
    volume.reset();
    volume_sampler.reset();
    repaint();
}


