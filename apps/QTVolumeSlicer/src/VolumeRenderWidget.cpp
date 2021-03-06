//
// Created by wyz on 2021/6/15.
//
#include "VolumeRenderWidget.hpp"
#include "camera.hpp"
#include <QMouseEvent>
#include <QPainter>
#include <VolumeSlicer/Common/frame.hpp>
#include <VolumeSlicer/Render/camera.hpp>
#include <VolumeSlicer/Render/transfer_function.hpp>
#include <VolumeSlicer/Render/volume_sampler.hpp>
#include <VolumeSlicer/Utils/utils.hpp>
VolumeRenderWidget::VolumeRenderWidget(QWidget *parent) {
//    initTest();
}

void VolumeRenderWidget::paintEvent(QPaintEvent *event) {
    if(!slicer || !dummy_slicer || ! raw_volume ) return;
    QPainter p(this);
    multi_volume_renderer->SetCamera(*base_camera);
    multi_volume_renderer->render();
    auto frame=multi_volume_renderer->GetFrame();
    QImage image(frame.data.data(),frame.width,frame.height,QImage::Format_RGBA8888,nullptr,nullptr);

//    std::cout<<"image "<<image.width()<<" "<<image.height()<<std::endl;
//    p.drawPixmap(0,0,QPixmap::fromImage(image.mirrored(false,true)));
    image.mirror(false,true);

    p.drawImage(0,0,image);
}

void VolumeRenderWidget::mouseMoveEvent(QMouseEvent *event) {
    if(!trackball_camera) return;
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
    if(!trackball_camera) return;
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
    if(!trackball_camera) return;
    trackball_camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         true,
                                         event->position().x(),
                                         event->position().y());
    event->accept();
    repaint();
}

void VolumeRenderWidget::mouseReleaseEvent(QMouseEvent *event) {
    if(!trackball_camera) return;
    trackball_camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         false,
                                         event->position().x(),
                                         event->position().y());
    event->accept();
    repaint();
}
void VolumeRenderWidget::loadVolume(const char * path,
                                    const std::array<uint32_t,3>& dim,
                                    const std::array<float,3>& space) {
    PrintCUDAMemInfo("before raw volume load");
    raw_volume=RawVolume::Load(path,VoxelType::UInt8,dim,space);
    PrintCUDAMemInfo("after raw volume load");
//    Slice slice;
//    slice.origin={raw_volume->GetVolumeDimX()/2.f,
//                  raw_volume->GetVolumeDimY()/2.f,
//                  raw_volume->GetVolumeDimZ()/2.f,1.f};
//    slice.right={1.f,0.f,0.f,0.f};
//    slice.up={0.f,1.f,-1.f,0.f};
//    slice.normal={0.f,1.f,1.f,0.f};
//    slice.n_pixels_width=400;
//    slice.n_pixels_height=400;
//    slice.voxel_per_pixel_height=1.f;
//    slice.voxel_per_pixel_width=1.f;
//    slicer=Slicer::CreateSlicer(slice);


//    multi_volume_renderer->SetSlicer(slicer);

    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
//    tf.points.emplace_back(114,std::array<double,4>{0.5,0.25,0.11,0.0});
//    tf.points.emplace_back(165,std::array<double,4>{0.5,0.25,0.11,0.6});
//    tf.points.emplace_back(216,std::array<double,4>{0.5,0.25,0.11,0.3});
    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});


    this->trackball_camera=std::make_unique<control::TrackBallCamera>(
            raw_volume->GetVolumeDimZ()*raw_volume->GetVolumeSpaceZ()/2.f,
            this->width(),this->height(),
            glm::vec3{raw_volume->GetVolumeDimX()*raw_volume->GetVolumeSpaceX()/2.f,
                      raw_volume->GetVolumeDimY()*raw_volume->GetVolumeSpaceY()/2.f,
                      raw_volume->GetVolumeDimZ()*raw_volume->GetVolumeSpaceZ()/2.f}
    );

    base_camera=std::make_unique<vs::Camera>();
    base_camera->pos={trackball_camera->getCameraPos().x,
                      trackball_camera->getCameraPos().y,
                      trackball_camera->getCameraPos().z};
    base_camera->up={0.f,1.f,0.f};
    base_camera->look_at={raw_volume->GetVolumeDimX()*raw_volume->GetVolumeSpaceX()/2.f,
                          raw_volume->GetVolumeDimY()*raw_volume->GetVolumeSpaceY()/2.f,
                          raw_volume->GetVolumeDimZ()*raw_volume->GetVolumeSpaceZ()/2.f};
    base_camera->zoom=60.f;
    base_camera->n=0.01f;
    base_camera->f=raw_volume->GetVolumeDimZ()*raw_volume->GetVolumeSpaceZ()*10.f;

    if(!multi_volume_renderer)
        multi_volume_renderer=CreateRenderer(this->width(),this->height());
    else
        multi_volume_renderer->resize(this->width(),this->height());
    multi_volume_renderer->SetVolume(raw_volume);
    multi_volume_renderer->SetTransferFunction(std::move(tf));
    multi_volume_renderer->SetCamera(*base_camera);
    multi_volume_renderer->SetVisible(true,true);
    PrintCUDAMemInfo("after multi_volume_renderer create");
    redraw();
}


void VolumeRenderWidget::setSlicer(const std::shared_ptr<Slicer> &slicer) {
    if(!slicer) return;
    this->slicer=slicer;
    auto slice=this->slicer->GetSlice();
    slice.origin={slice.origin[0]/64,slice.origin[1]/64,slice.origin[2]/64};
    slice.voxel_per_pixel_width/=64.f;
    slice.voxel_per_pixel_height/=64.f;
    this->dummy_slicer=Slicer::CreateSlicer(slice);
    multi_volume_renderer->SetSlicer(dummy_slicer);

}

void VolumeRenderWidget::redraw() {
    if(!slicer) return;
    setSlicer(this->slicer);
    repaint();
}

auto VolumeRenderWidget::getRawVolume() -> const std::shared_ptr<RawVolume> & {
    return this->raw_volume;
}

void VolumeRenderWidget::setVisible(bool volume, bool slice) {
    if(!multi_volume_renderer) return;
    multi_volume_renderer->SetVisible(volume,slice);
    redraw();
}

void VolumeRenderWidget::resetTransferFunc1D(float *data, int dim) {
    if(!multi_volume_renderer) return;
    multi_volume_renderer->SetTransferFunc1D(data,dim);
}


void VolumeRenderWidget::resizeEvent(QResizeEvent *event) {


    if(!multi_volume_renderer) return;

    multi_volume_renderer->resize(event->size().width(),event->size().height());
    this->trackball_camera->setScreenSize(event->size().width(),event->size().height());

//    QWidget::resizeEvent(event);
}

void VolumeRenderWidget::volumeLoaded() {

}

void VolumeRenderWidget::volumeClose() {
    spdlog::info("{0}.",__FUNCTION__ );
    slicer.reset();
    dummy_slicer.reset();
    raw_volume.reset();
//    multi_volume_renderer.reset();
    trackball_camera.reset();
    base_camera.reset();
    repaint();
}
