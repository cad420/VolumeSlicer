//
// Created by csh on 10/20/2021.
//

#include "VolumeRenderWidget.h"

VolumeRenderWidget::VolumeRenderWidget(QWidget *parent):QWidget(parent),
      frameHeight(700),
      frameWidth(700),
      moveSpeed(0.00002f),
      left_pressed(false),
      fps(30)
{
    setFocusPolicy(Qt::StrongFocus);

    space.resize(3);
    space[0] = 0.00032;
    space[1] = 0.00032;
    space[2] = 0.001;
}

void VolumeRenderWidget::loadVolume(const std::string& volume_config_file_path)
{
    sequenceNum = 0;

    int iGPU = 0;
    SetCUDACtx(iGPU);//used for cuda-renderer and volume
    realTimeRenderer = OpenGLCompVolumeRenderer::Create(frameWidth,frameHeight);

    volumeForRealTime = CompVolume::Load(volume_config_file_path.c_str());
    volumeForRealTime->SetSpaceX(space[0]);
    volumeForRealTime->SetSpaceY(space[1]);
    volumeForRealTime->SetSpaceZ(space[2]);
    realTimeRenderer->SetVolume(volumeForRealTime);
    realTimeRenderer->SetStep(0.00016,3000);

    //Camera
    baseCamera = std::make_unique<Camera>();
    fpsCamera = std::make_unique<control::FPSCamera>(glm::vec3( {4.9f,5.85f,5.93f}));
    updateCamera();

    //Render Policy
//    int max_lod = sizeof(CompRenderPolicy::lod_dist) / sizeof(CompRenderPolicy::lod_dist[0]);
//    std::vector<double> lod_policy(max_lod);
//    lod_policy[0] = std::numeric_limits<double>::max();

//    CompRenderPolicy policy;
//    for(int i = 0 ; i < max_lod ; i++){
//        policy.lod_dist[i] = lod_policy[i];
//    }
//    policy.cdf_value_file="chebyshev_dist_mouse_cdf_config.json";
//    realTimeRenderer->SetRenderPolicy(policy);

    //            tf.points.emplace_back(0,std::array<double,4>{0.1,0.0,0.0,0.0});
//            tf.points.emplace_back(74,std::array<double,4>{0.0,0.0,0.0,0.0});
//            tf.points.emplace_back(127,std::array<double,4>{0.75,0.75,0.75,0.6});
//            tf.points.emplace_back(128,std::array<double,4>{1.0,1.0,0.0,1.0});
//            tf.points.emplace_back(255,std::array<double,4>{1.0,0.0,0.0,1.0});

    //Transfer Function
//    TransferFunc tf;
//    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
//    tf.points.emplace_back(255,std::array<double,4>{1.0,1.0,1.0,1.0});
//
//    std::vector<float> info;
//    for(size_t i = 0;i < tf.points.size();i++){
//        info.push_back(std::floor(tf.points[i].key / 255.0f + 0.5f));
//        info.push_back(std::floor(tf.points[i].value[0] * 255.0f + 0.5f));
//        info.push_back(std::floor(tf.points[i].value[1] * 255.0f + 0.5f));
//        info.push_back(std::floor(tf.points[i].value[2] * 255.0f + 0.5f));
//        info.push_back(std::floor(tf.points[i].value[3] * 255.0f + 0.5f));
//    }
//    settingWidget->setTF(info.data(), tf.points.size());
//    realTimeRenderer->SetTransferFunc(std::move(tf));
}

void VolumeRenderWidget::paintEvent(QPaintEvent *event)
{
    event->accept();

    if(!realTimeRenderer || !volumeForRealTime) return;
    QPainter p(this);

    realTimeRenderer->render();
    auto& frame = realTimeRenderer->GetFrame();
    QImage image((uint8_t *)frame.data.data(),frame.width,frame.height,QImage::Format_RGBA8888,nullptr, nullptr);

    image.mirror(false,true);

    p.drawImage(0,0,image);
}

void VolumeRenderWidget::mousePressEvent(QMouseEvent *event)
{
    event->accept();
    if(event->button() == Qt::LeftButton){
        left_pressed = true;
        fpsCamera->processMouseButton(control::CameraDefinedMouseButton::Left, true, event->pos().x(),event->pos().y());
    }

}

void VolumeRenderWidget::mouseMoveEvent(QMouseEvent *event)
{
    if(!fpsCamera || !baseCamera) return;

    event->accept();
    if(left_pressed){
        fpsCamera->processMouseMove(event->position().x(), event->position().y());
        updateCamera();
        repaint();
    }
}

void VolumeRenderWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if(event->button() == Qt::LeftButton){
        left_pressed = false;
    }
}

void VolumeRenderWidget::wheelEvent(QWheelEvent *event)
{
    if(!fpsCamera || !baseCamera) return;

    event->accept();
    fpsCamera->processMouseScroll(event->angleDelta().y());
    updateCamera();
    repaint();
}

void VolumeRenderWidget::keyPressEvent(QKeyEvent *event)
{
    if(!fpsCamera || !baseCamera) return;

    auto key = event->key();
    switch(key){
    case 'W':{
        fpsCamera->processKeyEvent(control::CameraDefinedKey::Forward,moveSpeed);
        break;
    }
    case 'S': {
        fpsCamera->processKeyEvent(control::CameraDefinedKey::Backward,moveSpeed);
        break;
    }
    case 'A': {
        fpsCamera->processKeyEvent(control::CameraDefinedKey::Left, moveSpeed);
        break;
    }
    case 'D': {
        fpsCamera->processKeyEvent(control::CameraDefinedKey::Right,moveSpeed);
        break;
    }
    case 'Q': {
        fpsCamera->processKeyEvent(control::CameraDefinedKey::Up, moveSpeed);
        break;
    }
    case 'E': {
        fpsCamera->processKeyEvent(control::CameraDefinedKey::Bottom,moveSpeed);
        break;
    }
        default:return;
    }
    updateCamera();
    repaint();
}

void VolumeRenderWidget::updateCamera()
{
    if(!fpsCamera || !baseCamera) return;

    auto pos = fpsCamera->getCameraPos();
    baseCamera->pos = {pos[0],pos[1],pos[2]};
    baseCamera->zoom = fpsCamera->getZoom();
    auto lookAt = fpsCamera->getCameraLookAt();
    baseCamera->look_at = {lookAt[0], lookAt[1],lookAt[2]};
    auto up = fpsCamera->getCameraUp();
    baseCamera->up = {up[0], up[1], up[2]};
    auto right = fpsCamera->getCameraRight();
    baseCamera->right = {right[0], right[1], right[2]};

    realTimeRenderer->SetCamera(*baseCamera);
}

void VolumeRenderWidget::setRenderPolicy(const float* renderPolicy,int num)
{
    if(!realTimeRenderer )
        return;

    int max_lod = sizeof(CompRenderPolicy::lod_dist) / sizeof(CompRenderPolicy::lod_dist[0]);

    CompRenderPolicy policy;
    int i;
    for (i = 0; i < num && i < max_lod; i++)
    {
        policy.lod_dist[i] = renderPolicy[i];
    }
    policy.cdf_value_file="chebyshev_dist_mouse_cdf_config.json";

    realTimeRenderer->SetRenderPolicy(policy);
    repaint();
}

//void VolumeRenderWidget::setTransferFunction(float* tfData, int* index,int num, bool type)
//{
//    if(!realTimeRenderer || !offScreenRenderer)
//        return;
//
//    TransferFunc tf;
//    int i = 0;
//    for(;i < num;i++){
//        tf.points.emplace_back(index[i],std::array<double,4>{tfData[i * 4 + 0],tfData[i * 4 + 1],tfData[i * 4 + 2],tfData[i * 4 + 3]});
//    }
//    if(type == false){
//        realTimeRenderer->SetTransferFunc(std::move(tf));
//        repaint();
//    }
//    else
//        offScreenRenderer->SetTransferFunc(std::move(tf));
//}

void VolumeRenderWidget::setTransferFunction(const float* tfData, int num)
{
    if(!realTimeRenderer)
        return;

    TransferFunc tf;
    int i = 0;
    for(;i < num;i++){
        tf.points.emplace_back(i,std::array<double,4>{tfData[i * 4 + 0],tfData[i * 4 + 1],tfData[i * 4 + 2],tfData[i * 4 + 3]});
    }

    realTimeRenderer->SetTransferFunc(std::move(tf));
    repaint();
}

std::shared_ptr<CompVolume> VolumeRenderWidget::getVolumeForRealTime()
{
    return volumeForRealTime;
}

void VolumeRenderWidget::volumeClosed()
{
    spdlog::info("{0}.",__FUNCTION__ );
    realTimeRenderer.reset();
    volumeForRealTime.reset();
    baseCamera.reset();
    fpsCamera.reset();
    cameraSequence.clear();

    repaint();
}

//int VolumeRenderWidget::getFPS(){
//    return fps;
//}

void VolumeRenderWidget::setWidget(SettingWidget* in_settingWidget)
{
    settingWidget = in_settingWidget;
//    offScreenRenderSettingWidget = in_offScreenRenderSettingWidget;
}

void VolumeRenderWidget::setFPS(const int in_fps){
    fps = in_fps;
}

void VolumeRenderWidget::startRecording(){
    cameraSequence.clear();
    timer = new QTimer;
    timer->setInterval(std::floor(1.0f/fps * 1000 + 0.5));
    connect(timer,&QTimer::timeout, [this](){
        auto camera = std::make_unique<Camera>();
        camera->up = baseCamera->up;
        camera->pos = baseCamera->pos;
        camera->look_at = baseCamera->look_at;
        camera->right = baseCamera->right;
        camera->zoom = baseCamera->zoom;

        this->cameraSequence.emplace_back(std::move(camera));
    });
    timer->start();
}

void VolumeRenderWidget::stopRecording(){
    timer->stop();

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save TF1D"), QString::fromStdString(curCameraFile), tr("Camera Sequence(*.json);;All Files (*)"));

//    auto time = std::time(0);
//    auto tm = localtime(&time);
//    std::string tmStr = std::to_string(tm->tm_mon)+"-"+std::to_string(tm->tm_mday)+" "+std::to_string(tm->tm_hour)+"h"+std::to_string(tm->tm_min)+"m"+std::to_string(tm->tm_sec)+"s";
//    std::string fileName = tmStr + " camera_sequence_"+  std::to_string(sequenceNum + 1 )+  ".json";

    nlohmann::json j;
    j["fps"] = fps;
    j["frame_count"] = cameraSequence.size();
    j["property"] = {"zoom","pos","look_at","up","right"};
    for(int i=0;i < cameraSequence.size();i++){
        std::string idx="frame_"+std::to_string(i+1);
        j[idx]={
            cameraSequence[i]->zoom,
            cameraSequence[i]->pos,
            cameraSequence[i]->look_at,
            cameraSequence[i]->up,
            cameraSequence[i]->right
        };
    }

    std::ofstream out(fileName.toStdString());
    out << j <<std::endl;
    out.close();

    curCameraFile = QFileInfo(fileName).absolutePath().toStdString();

    settingWidget->setCameraName(fileName.toStdString());

//    std::vector<float> tf;
//    tf.resize(256*4);
//    settingWidget->getTransferFunc(tf.data());
//    for(int i=0;i < tf.size()/4;i++){
//        std::string idx=std::to_string(i);
//        j["tf"][idx]={
//            tf[i*4+0],
//            tf[i*4+1],
//            tf[i*4+2],
//            tf[i*4+3]
//        };
//    }

    sequenceNum++;
}

void VolumeRenderWidget::draw(){
    repaint();
}