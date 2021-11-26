//
// Created by csh on 10/20/2021.
//

#include "RealTimeVolumeRenderWidget.hpp"

RealTimeVolumeRenderWidget::RealTimeVolumeRenderWidget(QWidget *parent):QWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);


}



void RealTimeVolumeRenderWidget::paintEvent(QPaintEvent *event)
{
    if(!real_time_renderer) return;
    updateCamera();

    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    real_time_renderer->render(true);
    auto& image = real_time_renderer->GetImage();
    QImage img = QImage(reinterpret_cast<const uint8_t*>(image.GetData()),image.Width(),image.Height(),QImage::Format_RGBA8888,nullptr,nullptr);
    img.mirror(false,true);
    p.drawImage(0,0,img);
}

void RealTimeVolumeRenderWidget::mousePressEvent(QMouseEvent *event)
{
    if(!real_time_renderer) return;
    left_mouse_press = true;
    control_camera->processMouseButton(control::CameraDefinedMouseButton::Left,true,event->pos().x(),event->pos().y());
    event->accept();
    repaint();
}

void RealTimeVolumeRenderWidget::mouseMoveEvent(QMouseEvent *event)
{
    if(!real_time_renderer) return;
    if(!left_mouse_press) return;
    control_camera->processMouseMove(event->pos().x(),event->pos().y());
    event->accept();
    repaint();
}

void RealTimeVolumeRenderWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if(!real_time_renderer) return;
    left_mouse_press = false;
    control_camera->processMouseButton(control::CameraDefinedMouseButton::Left,false,event->pos().x(),event->pos().y());
    event->accept();
    repaint();
}

void RealTimeVolumeRenderWidget::wheelEvent(QWheelEvent *event)
{
    if(!real_time_renderer) return;
    control_camera->processMouseScroll(event->angleDelta().y());
    event->accept();
    repaint();
}

void RealTimeVolumeRenderWidget::keyPressEvent(QKeyEvent *event)
{
    static float moveSpeed = 0.0001;
    if(!real_time_renderer) return;
    auto key = event->key();
    switch(key){
    case 'W':{
        control_camera->processKeyEvent(control::CameraDefinedKey::Forward,moveSpeed);
        break;
    }
    case 'S': {
        control_camera->processKeyEvent(control::CameraDefinedKey::Backward,moveSpeed);
        break;
    }
    case 'A': {
        control_camera->processKeyEvent(control::CameraDefinedKey::Left, moveSpeed);
        break;
    }
    case 'D': {
        control_camera->processKeyEvent(control::CameraDefinedKey::Right,moveSpeed);
        break;
    }
    case 'Q': {
        control_camera->processKeyEvent(control::CameraDefinedKey::Up, moveSpeed);
        break;
    }
    case 'E': {
        control_camera->processKeyEvent(control::CameraDefinedKey::Bottom,moveSpeed);
        break;
    }
    case 'F':{
        moveSpeed *= 2;
        break;
    }
    case 'G':{
        moveSpeed /=2;
        break;
    }
    default:return;
    }
    event->accept();
    repaint();
}
void RealTimeVolumeRenderWidget::loadVolume(const std::string &path)
{
    comp_volume = CompVolume::Load(path.c_str());

    real_time_renderer = OpenGLCompVolumeRenderer::Create(this->width(),this->height());
    real_time_renderer->SetVolume(comp_volume);

    //default policy for real-time renderer
    {
        CompRenderPolicy policy;
        policy.lod_dist[0]=0.3;
        policy.lod_dist[1]=0.6;
        policy.lod_dist[2]=1.2;
        policy.lod_dist[3]=2.4;
        policy.lod_dist[4]=4.8;
        policy.lod_dist[5]=9.6;
        policy.lod_dist[6]=std::numeric_limits<double>::max();
        real_time_renderer->SetRenderPolicy(policy);
    }
    //default tf
    {
        TransferFunc tf;
        tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
        tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
        real_time_renderer->SetTransferFunc(std::move(tf));
    }


    control_camera = std::make_unique<control::FPSCamera>(glm::vec3{4.90f,5.858f,7.23f});

    emit volumeLoaded(comp_volume);

    repaint();
}
void RealTimeVolumeRenderWidget::closeVolume()
{
    if(recording_camera){
        stopRecording();
    }

    comp_volume.reset();
    real_time_renderer.reset();
    recording_camera = false;

    emit volumeClosed();

    repaint();
}
void RealTimeVolumeRenderWidget::startRecording()
{
    if(!real_time_renderer) return;
    if(recording_camera){
        LOG_ERROR("startRecording error: recoding is on, should stop recording first");
        return;
    }
    recording_camera = true;
    emit recordingStart();
}
void RealTimeVolumeRenderWidget::stopRecording()
{
    if(!real_time_renderer) return;
    if(!recording_camera){
        LOG_ERROR("stopRecording error: recoding is off already");
        return;
    }
    recording_camera = false;
    emit recordingFinish(std::move(this->recording_cameras));
    clearRecordCameras();
}
void RealTimeVolumeRenderWidget::updateCamera()
{
    if(!real_time_renderer) return;

    Camera camera;
    camera.zoom = control_camera->getZoom();
    auto pos = control_camera->getCameraPos();
    camera.pos = {pos.x,pos.y,pos.z};
    auto lookat = control_camera->getCameraLookAt();
    camera.look_at = {lookat.x,lookat.y,lookat.z};
    auto up = control_camera->getCameraUp();
    camera.up = {up.x,up.y,up.z};
    auto right = control_camera->getCameraRight();
    camera.right = {right.x,right.y,right.z};

    real_time_renderer->SetCamera(camera);

    if(recording_camera){
        recordCurrentCamera(camera);
    }

}
void RealTimeVolumeRenderWidget::recordCurrentCamera(const Camera& camera)
{
    assert(recording_camera);

    recording_cameras.emplace_back(camera);
}
void RealTimeVolumeRenderWidget::clearRecordCameras()
{
    if(recording_camera){
        LOG_ERROR("clearRecordCameras still recording_camera is true");
        return ;
    }
    recording_cameras.clear();
    LOG_INFO("clear recording cameras");
}
void RealTimeVolumeRenderWidget::updateTransferFunc(float *data, int dim)
{
    if(!real_time_renderer) return;
    TransferFunc tf;
    for(int i=0;i<dim;i++){
        tf.points.emplace_back(i,std::array<double,4>{
                                      (double)data[i*4+0],(double)data[i*4+1],
                                      (double)data[i*4+2],(double)data[i*4+3]});
    }
    real_time_renderer->SetTransferFunc(std::move(tf));
    repaint();
}
void RealTimeVolumeRenderWidget::updateSteps(int steps)
{
    static auto space_x = comp_volume->GetVolumeSpaceX();
    static auto space_y = comp_volume->GetVolumeSpaceY();
    static auto space_z = comp_volume->GetVolumeSpaceZ();
    static auto base_space = std::min({space_x,space_y,space_z});
    if(!real_time_renderer) return;
    real_time_renderer->SetStep(base_space*0.5,steps);
    repaint();
}
