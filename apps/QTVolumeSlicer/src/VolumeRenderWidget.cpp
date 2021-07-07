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
#include "camera.hpp"
VolumeRenderWidget::VolumeRenderWidget(QWidget *parent) {
    initTest();

}

void VolumeRenderWidget::paintEvent(QPaintEvent *event) {
    QPainter p(this);
    multi_volume_renderer->SetCamera(*base_camera);
    multi_volume_renderer->render();
    auto frame=multi_volume_renderer->GetFrame();
    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format_RGBA8888,nullptr,nullptr);

    std::cout<<"image "<<image.width()<<" "<<image.height()<<std::endl;
//    p.drawPixmap(0,0,QPixmap::fromImage(image.mirrored(false,true)));
    image.mirror(false,true);

    p.drawImage(0,0,image);
}

void VolumeRenderWidget::mouseMoveEvent(QMouseEvent *event) {
    trackball_camera->processMouseMove(event->pos().x(),event->pos().y());
    auto pos=trackball_camera->getCameraPos();
    auto lookat=trackball_camera->getCameraLookAt();
    auto up=trackball_camera->getCameraUp();
    this->base_camera->pos={pos.x,pos.y,pos.z};
    this->base_camera->up={up.x,up.y,up.z};
    this->base_camera->look_at={lookat.x,lookat.y,lookat.z};
    event->accept();
    repaint();
}

void VolumeRenderWidget::wheelEvent(QWheelEvent *event) {
    setFocus();
    trackball_camera->processMouseScroll(event->angleDelta().y());
    auto pos=trackball_camera->getCameraPos();
    auto lookat=trackball_camera->getCameraLookAt();
    auto up=trackball_camera->getCameraUp();
    this->base_camera->pos={pos.x,pos.y,pos.z};
    this->base_camera->up={up.x,up.y,up.z};
    this->base_camera->look_at={lookat.x,lookat.y,lookat.z};
    event->accept();
    repaint();
}

void VolumeRenderWidget::mousePressEvent(QMouseEvent *event) {
    setFocus();
    trackball_camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         true,
                                         event->position().x(),
                                         event->position().y());
    event->accept();
    repaint();
}

void VolumeRenderWidget::mouseReleaseEvent(QMouseEvent *event) {
    trackball_camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         false,
                                         event->position().x(),
                                         event->position().y());
    event->accept();
    repaint();
}

void VolumeRenderWidget::initTest() {
    Slice slice;
    slice.origin={128.f,128.f,128.f,1.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,1.f,-1.f,0.f};
    slice.normal={0.f,1.f,1.f,0.f};
    slice.n_pixels_width=400;
    slice.n_pixels_height=400;
    slice.voxel_per_pixel_height=1.f;
    slice.voxel_per_pixel_width=1.f;
    slicer=Slicer::CreateSlicer(slice);
    raw_volume=RawVolume::Load("E:\\mouse_23389_29581_10296_512_2_lod6/mouselod6_366_463_161_uint8.raw",
                               VoxelType::UInt8,
                               {366,463,161},
                               {0.01f,0.01f,0.03f});
    multi_volume_renderer=CreateRenderer(slice.n_pixels_width,slice.n_pixels_height);
    multi_volume_renderer->SetVolume(raw_volume);
//    multi_volume_renderer->SetSlicer(slicer);

    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(114,std::array<double,4>{0.5,0.25,0.11,0.0});
    tf.points.emplace_back(165,std::array<double,4>{0.5,0.25,0.11,0.6});
    tf.points.emplace_back(216,std::array<double,4>{0.5,0.25,0.11,0.3});
    tf.points.emplace_back(255,std::array<double,4>{0.0,0.0,0.0,0.0});
    multi_volume_renderer->SetTransferFunction(std::move(tf));

    base_camera=std::make_unique<vs::Camera>();
    base_camera->pos={1.83f,2.315f,2.415f+6.f};
    base_camera->up={0.f,1.f,0.f};
    base_camera->look_at={1.83f,2.315f,2.415f};
    base_camera->zoom=60.f;
    base_camera->n=0.01f;
    base_camera->f=20.f;
    multi_volume_renderer->SetCamera(*base_camera);
    multi_volume_renderer->SetVisible(false,true);
    this->trackball_camera=std::make_unique<control::TrackBallCamera>(
                3.f,slice.n_pixels_width,slice.n_pixels_height,
                glm::vec3{1.83f,2.315f,2.415f}
            );

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

auto VolumeRenderWidget::getRawVolume() -> const std::shared_ptr<RawVolume> & {
    return this->raw_volume;
}
