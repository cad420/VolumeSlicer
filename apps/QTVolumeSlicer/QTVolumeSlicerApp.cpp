//
// Created by wyz on 2021/6/11.
//
#include"QTVolumeSlicerApp.hpp"
#include "VolumeRenderWidget.hpp"
#include <QMenuBar>
#include <iostream>
#include <QPainter>
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
    multi_renderer->render();
    auto frame=multi_renderer->GetFrame();
    const uchar* data=frame.data.data();
    QImage image(data,frame.width,frame.height,QImage::Format::Format_RGBA8888,nullptr,nullptr);
//    QImage image(QString(ICONS_PATH)+"open.png");

    auto pix=QPixmap::fromImage(image.mirrored(false,true));
    auto w=pix.width();
    p.drawPixmap(0,0,pix);
}

void VolumeSlicerMainWindow::initTest() {
    std::cout<<__FUNCTION__ <<std::endl;
    raw_volume=RawVolume::Load("C:\\Users\\wyz\\projects\\VolumeSlicer\\test_data\\aneurism_256_256_256_uint8.raw",VoxelType::UInt8,{256,256,256},{0.01f,0.01f,0.01f});
    multi_renderer=CreateRenderer(1200,900);
    multi_renderer->SetVolume(raw_volume);
    Slice slice;
    slice.origin={128.f,128.f,128.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,1.f,-1.f,0.f};
    slice.normal={0.f,1.f,1.f,0.f};
    slice.n_pixels_height=400;
    slice.n_pixels_width=300;
    slice.voxel_per_pixel_height=1.f;
    slice.voxel_per_pixel_width=1.f;
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

}
