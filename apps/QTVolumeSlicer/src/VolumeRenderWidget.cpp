//
// Created by wyz on 2021/6/15.
//
#include"VolumeRenderWidget.hpp"
#include <VolumeSlicer/transfer_function.hpp>
#include <VolumeSlicer/camera.hpp>
#include <VolumeSlicer/frame.hpp>
#include <VolumeSlicer/utils.hpp>
#include <VolumeSlicer/volume_sampler.hpp>
#include <QPainter>
#include <QMouseEvent>
VolumeRenderWidget::VolumeRenderWidget(QWidget *parent) {
    initTest();

}

void VolumeRenderWidget::paintEvent(QPaintEvent *event) {
    QPainter p(this);

    multi_volume_renderer->render();
    auto frame=multi_volume_renderer->GetFrame();
    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format::Format_RGBA8888,nullptr,nullptr);
    p.drawPixmap(0,0,QPixmap::fromImage(image.mirrored(false,true)));

}

void VolumeRenderWidget::mouseMoveEvent(QMouseEvent *event) {

    event->accept();

}

void VolumeRenderWidget::wheelEvent(QWheelEvent *event) {

    event->accept();

}

void VolumeRenderWidget::mousePressEvent(QMouseEvent *event) {

    event->accept();
    repaint();
}

void VolumeRenderWidget::mouseReleaseEvent(QMouseEvent *event) {

    event->accept();

}

void VolumeRenderWidget::initTest() {
    Slice slice;
    slice.origin={128.f,128.f,128.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,1.f,-1.f,0.f};
    slice.normal={0.f,1.f,1.f,0.f};
    slice.n_pixels_width=500;
    slice.n_pixels_height=500;
    slice.voxel_per_pixel_height=1.f;
    slice.voxel_per_pixel_width=1.f;
    slicer=Slicer::CreateSlicer(slice);
    raw_volume=RawVolume::Load("E:\\mouse_23389_29581_10296_512_2_lod6/mouselod6_366_463_161_uint8.raw",
                               VoxelType::UInt8,
                               {366,463,161},
                               {0.01f,0.01f,0.03f});
    multi_volume_renderer=CreateRenderer(500,500);
    multi_volume_renderer->SetVolume(raw_volume);
//    multi_volume_renderer->SetSlicer(slicer);

    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(114,std::array<double,4>{0.5,0.25,0.11,0.0});
    tf.points.emplace_back(165,std::array<double,4>{0.5,0.25,0.11,0.6});
    tf.points.emplace_back(216,std::array<double,4>{0.5,0.25,0.11,0.3});
    tf.points.emplace_back(255,std::array<double,4>{0.0,0.0,0.0,0.0});
    multi_volume_renderer->SetTransferFunction(std::move(tf));

    Camera camera;
    camera.pos={1.68f,2.28f,8.f};
    camera.up={0.f,1.f,0.f};
    camera.front={0.f,0.f,-1.f};
    camera.zoom=60.f;
    camera.n=0.01f;
    camera.f=10.f;
    multi_volume_renderer->SetCamera(camera);
    multi_volume_renderer->SetVisible(false,true);

}

void VolumeRenderWidget::setSlicer(const std::shared_ptr<Slicer> &slicer) {
    this->slicer=slicer;
    auto slice=this->slicer->GetSlice();
    slice.origin={slice.origin[0]/64,slice.origin[1]/64,slice.origin[2]/64};
    slice.voxel_per_pixel_width/=64.f;
    slice.voxel_per_pixel_height/=64.f;
    this->dummy_slicer=Slicer::CreateSlicer(slice);
    multi_volume_renderer->SetSlicer(dummy_slicer);
    multi_volume_renderer->SetVisible(true,true);

}

void VolumeRenderWidget::redraw() {
    setSlicer(this->slicer);
    repaint();
}
