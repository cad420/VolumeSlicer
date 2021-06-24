//
// Created by wyz on 2021/6/11.
//
#include"QTVolumeSlicerApp.hpp"
#include "VolumeRenderWidget.hpp"
#include <QMenuBar>
#include <iostream>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QApplication>
#include "global.hpp"
VolumeSlicerMainWindow::VolumeSlicerMainWindow(QWidget *parent)
:QMainWindow(parent)
{
    setWindowTitle("VolumeSlicer");
    resize(1600,1080);
    createActions();
    createMenu();
    initTest();
}

void VolumeSlicerMainWindow::createMenu() {
    m_file_menu=menuBar()->addMenu("File");
    m_file_open_menu=m_file_menu->addMenu("Open");
    m_file_open_menu->addAction("Raw",this,[](){
       std::cout<<"Open Raw"<<std::endl;
    });
    m_file_open_menu->addAction("Comp",this,[](){
        std::cout<<"Open Comp"<<std::endl;
    });
    m_file_menu->addSeparator();
    m_file_menu->addAction(tr("Close"),this,[](){
        std::cout<<"Close"<<std::endl;
    });


}

void VolumeSlicerMainWindow::createActions() {

}

void VolumeSlicerMainWindow::paintEvent(QPaintEvent* event) {
    std::cout<<__FUNCTION__ <<std::endl;
    QPainter p(this);

//    multi_renderer->render();
//    auto frame=multi_renderer->GetFrame();
    Frame frame;
    frame.width=slicer->GetImageW();
    frame.height=slicer->GetImageH();
    frame.channels=1;
    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
//    volume_sampler->Sample(slicer->GetSlice(),frame.data.data());

    comp_volume_sampler->Sample(slicer->GetSlice(),frame.data.data());

    const uchar* data=frame.data.data();
    QImage image(data,frame.width,frame.height,QImage::Format::Format_Grayscale8,nullptr,nullptr);
//    QImage image(QString(ICONS_PATH)+"open.png");

    auto pix=QPixmap::fromImage(image.mirrored(false,false));
    auto w=pix.width();
    p.drawPixmap(0,0,pix);
}
void VolumeSlicerMainWindow::drawVolume() {

}

void VolumeSlicerMainWindow::wheelEvent(QWheelEvent *event) {

    auto angle_delta=event->angleDelta();
    if((QApplication::keyboardModifiers() == Qt::ControlModifier)){
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
}
void VolumeSlicerMainWindow::mousePressEvent(QMouseEvent *event) {

    left_mouse_button_pressed=true;
    last_pos=event->pos();
    event->accept();
    repaint();
}
void VolumeSlicerMainWindow::mouseMoveEvent(QMouseEvent *event) {
    if(left_mouse_button_pressed){
        auto pos=event->pos();
        auto d=pos-last_pos;
        last_pos=pos;
        std::cout<<d.x()<<" "<<d.y()<<std::endl;
        slicer->MoveInPlane(d.x(),d.y());
    }
    event->accept();
    repaint();
}

void VolumeSlicerMainWindow::mouseReleaseEvent(QMouseEvent *event) {
    left_mouse_button_pressed=false;
    event->accept();
    repaint();
}
void VolumeSlicerMainWindow::keyPressEvent(QKeyEvent *event) {

    event->accept();
    repaint();
}
void VolumeSlicerMainWindow::initTest() {
    std::cout<<__FUNCTION__ <<std::endl;
    raw_volume=RawVolume::Load("C:\\Users\\wyz\\projects\\VolumeSlicer\\test_data\\aneurism_256_256_256_uint8.raw",VoxelType::UInt8,{256,256,256},{0.01f,0.01f,0.01f});
    multi_renderer=CreateRenderer(1200,900);
    multi_renderer->SetVolume(raw_volume);
    Slice slice;
    slice.origin={9765.f,8434.f,4541.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,1.f,0.f,0.f};
    slice.normal={0.f,0.f,1.f,0.f};
    slice.n_pixels_width=1200;
    slice.n_pixels_height=900;
    slice.voxel_per_pixel_height=2.f;
    slice.voxel_per_pixel_width=2.f;
    slicer=Slicer::CreateSlicer(slice);
    multi_renderer->SetSlicer(slicer);
    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(114,std::array<double,4>{0.5,0.25,0.11,0.0});
    tf.points.emplace_back(165,std::array<double,4>{0.5,0.25,0.11,0.6});
    tf.points.emplace_back(216,std::array<double,4>{0.5,0.25,0.11,0.3});
    tf.points.emplace_back(255,std::array<double,4>{0.0,0.0,0.0,0.0});
    multi_renderer->SetTransferFunction(std::move(tf));
    Camera camera;
    camera.pos={1.28f,1.28f,5.2f};
    camera.up={0.f,1.f,0.f};
    camera.front={0.f,0.f,-1.f};
    camera.zoom=60.f;
    camera.n=0.01f;
    camera.f=10.f;
    multi_renderer->SetCamera(camera);
    multi_renderer->SetVisible(true,true);
    volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);


    this->comp_volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
    auto block_length=comp_volume->GetBlockLength();
    std::cout<<"block length: "<<block_length[0]<<" "<<block_length[1]<<std::endl;
    auto block_dim=comp_volume->GetBlockDim(0);
    std::cout<<"block dim: "<<block_dim[0]<<" "<<block_dim[1]<<" "<<block_dim[2]<<std::endl;

    comp_volume_sampler=VolumeSampler::CreateVolumeSampler(comp_volume);
    comp_volume->SetSpaceX(0.01f);
    comp_volume->SetSpaceY(0.01f);
    comp_volume->SetSpaceZ(0.01f);
}
