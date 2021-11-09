//
// Created by csh on 10/20/2021.
//

#include "OffScreenRenderSetting.h"

OffScreenRenderSettingWidget::OffScreenRenderSettingWidget(VolumeRenderWidget* volumeRenderWidget, SettingWidget* settingWidget, QWidget* parent):volumeRenderWidget(volumeRenderWidget),
      settingWidget(settingWidget),
    frameHeight(700),
    frameWidth(700),
    QWidget(parent)
{
    space.resize(3);
    space[0]=0.00032f;
    space[1]=0.00032f;
    space[2]=0.001f;

    auto widgetLayout = new QVBoxLayout;

//    auto volumeSpaceSettingLayout = new QVBoxLayout;
//    auto volumeSpaceSetting = new QGroupBox("Volume Space");
//    volumeSpaceSetting->setLayout(volumeSpaceSettingLayout);
//    volumeSpaceSetting->setFixedHeight(150);
//
//    auto spaceXLayout = new QHBoxLayout;
//    auto spaceXLabel = new QLabel("x");
//    spaceXLabel->setAlignment(Qt::AlignHCenter);
//    space_x = new QDoubleSpinBox();
//    space_x->setRange(0.f, 0.0005f);
//    space_x->setDecimals(5);
//    space_x->setValue(0.00032f);
//    space_x->setSingleStep(0.00001);
//    spaceXLayout->addWidget(spaceXLabel);
//    spaceXLayout->addWidget(space_x);
//    volumeSpaceSettingLayout->addLayout(spaceXLayout);
//
//    auto spaceYLayout = new QHBoxLayout;
//    auto spaceYLabel = new QLabel("y");
//    spaceYLabel->setAlignment(Qt::AlignHCenter);
//    space_y = new QDoubleSpinBox();
//    space_y->setRange(0.f, 0.0005f);
//    space_y->setDecimals(5);
//    space_y->setValue(0.00032f);
//    space_y->setSingleStep(0.00001);
//    spaceYLayout->addWidget(spaceYLabel);
//    spaceYLayout->addWidget(space_y);
//    volumeSpaceSettingLayout->addLayout(spaceYLayout);
//
//    auto spaceZLayout = new QHBoxLayout;
//    auto spaceZLabel = new QLabel("z");
//    spaceZLabel->setAlignment(Qt::AlignHCenter);
//    space_z = new QDoubleSpinBox();
//    space_z->setRange(0.f, 0.003f);
//    space_z->setDecimals(4);
//    space_z->setSingleStep(0.0001);
//    space_z->setValue(0.001f);
//    spaceZLayout->addWidget(spaceZLabel);
//    spaceZLayout->addWidget(space_z);
//    volumeSpaceSettingLayout->addLayout(spaceZLayout);
//    volumeSpaceSetting->setVisible(false);
//    widgetLayout->addWidget(volumeSpaceSetting);

    auto renderQualityGroupBox = new QGroupBox("render quality");
    auto renderQualityLayout = new QHBoxLayout;
    renderQualityGroupBox->setLayout(renderQualityLayout);
    auto lowLabel = new QLabel("low");
    lowLabel->setFixedWidth(70);
    auto highLabel = new QLabel("high");
    highLabel->setFixedWidth(70);
    qualitySlider = new QSlider(Qt::Horizontal);
    qualitySlider->setRange(0,10);
    renderQualityLayout->addWidget(lowLabel);
    renderQualityLayout->addWidget(qualitySlider);
    renderQualityLayout->addWidget(highLabel);
    widgetLayout->addWidget(renderQualityGroupBox);

    auto cameraButtonLayout = new QVBoxLayout;
    loadCameraButton = new QPushButton("load camera sequence");
    editCameraButton = new QPushButton("edit camera sequence");
    cameraFileLabel =new QLabel("Camera sequence file not loaded");
    cameraButtonLayout->addWidget(loadCameraButton);
    cameraButtonLayout->addWidget(editCameraButton);
    cameraButtonLayout->addWidget(cameraFileLabel);
    widgetLayout->addLayout(cameraButtonLayout);

//    auto renderPolicyLayout = new QVBoxLayout;
//    render_policy_editor = new RenderPolicyEditor(this);
//    render_policy_editor->setFixedHeight(150);
//    renderPolicy.resize(10);
//    renderPolicyLayout->addWidget(render_policy_editor);
//    copyButton = new QPushButton("copy render policy from real time render setting");
//    renderPolicyLayout->addWidget(copyButton);
//    widgetLayout->addLayout(renderPolicyLayout);

    auto videoSettingLayout = new QVBoxLayout;
    auto videoSettingGroupBox = new QGroupBox("Video Setting");
    videoSettingGroupBox->setLayout(videoSettingLayout);
    videoSettingGroupBox->setFixedHeight(100);

    auto checkBoxLayout = new QHBoxLayout;
    auto checkBoxLabel = new QLabel("save image");
    checkBoxLabel->setAlignment(Qt::AlignHCenter);
    imageSaving = new QCheckBox;
    checkBoxLayout->addWidget(checkBoxLabel);
    checkBoxLayout->addWidget(imageSaving);
    videoSettingLayout->addLayout(checkBoxLayout);

//    auto fpsLayout = new QHBoxLayout;
//    auto fpsLabel = new QLabel("fps");
//    fpsLabel->setAlignment(Qt::AlignHCenter);
//    fps = new QSpinBox();
//    fps->setRange(15, 50);
//    fps->setValue(30);
//    fps->setSingleStep(1);
//    fpsLayout->addWidget(fpsLabel);
//    fpsLayout->addWidget(fps);
//    videoSettingLayout->addLayout(fpsLayout);

    auto fileNameLayout = new QHBoxLayout;
    auto fileNameLabel = new QLabel("file name");
    fileNameLabel->setAlignment(Qt::AlignHCenter);
    name = new QLineEdit();
    name->setFixedWidth(180);
    name->setText("result");
    fileNameLayout->addWidget(fileNameLabel);
    fileNameLayout->addWidget(name);
    videoSettingLayout->addLayout(fileNameLayout);

    widgetLayout->addWidget(videoSettingGroupBox);

    auto buttonLayout = new QHBoxLayout;
    startButton = new QPushButton("start rendering");
    startButton->setEnabled(false);
    buttonLayout->addWidget(startButton);
    saveButton = new QPushButton("save setting file");
    saveButton->setEnabled(false);
    buttonLayout->addWidget(saveButton);
//    stopButton = new QPushButton("stop and render");
//    stopButton->setEnabled(false);
//    buttonLayout->addWidget(stopButton);
    widgetLayout->addLayout(buttonLayout);

    this->setLayout(widgetLayout);

    initCameraDialog();

    connect(loadCameraButton,&QPushButton::clicked,[this](){
      auto camera_sequence_config = QFileDialog::getOpenFileName(this,
                                   QStringLiteral("OpenFile"),
                                   QStringLiteral("."),
                                   QStringLiteral("config files(*.json)")
      ).toStdString();

      std::ifstream in;
      in.open(camera_sequence_config);
      if(!in.is_open()){
          spdlog::error("Open camera sequence config file failed");
          throw std::runtime_error("Open camera sequence config file failed");
      }

      nlohmann::json j;
      in >> j;
      int frame_count;
      if(j.find("frame_count") != j.end()){
          frame_count = j.at("frame_count");
      }
      else{
          frame_count = 0;
          spdlog::error("Not provide frame_count, use default frame_count(0)");
      }
      if(j.find("property") != j.end()){
          auto property = j.at("property");
          bool b0 = property[0] == "zoom";
          bool b1 = property[1] == "pos";
          bool b2 = property[2] == "look_at";
          bool b3 = property[3] == "up";
          bool b4 = property[4] == "right";
          if(!b0 || !b1 || !b2 || !b3 || !b4){
              spdlog::error("Camera property order or name not correct");
              throw std::runtime_error("Camera property order or name not correct");
          }
      }

      this->cameraList->clear();
      this->cameraSequence.clear();
      this->cameraSequence.reserve(frame_count);
      for(int i=0;i<frame_count;i++){
          auto frame_idx = "frame_"+std::to_string(i+1);
          auto frame_camera = j.at(frame_idx);
          auto camera = std::make_unique<Camera>();
          camera->zoom = frame_camera[0];
          camera->pos = {frame_camera[1][0],frame_camera[1][1],frame_camera[1][2]};
          camera->look_at = {frame_camera[2][0],frame_camera[2][1],frame_camera[2][2]};
          camera->up = {frame_camera[3][0],frame_camera[3][1],frame_camera[3][2]};
          camera->right = {frame_camera[4][0],frame_camera[4][1],frame_camera[4][2]};

          this->cameraList->addItem(frame_idx.c_str());

          this->cameraSequence.emplace_back(std::move(camera));
      }
      //std::cout <<"frame count:"<<this->cameraSequence.size()<<std::endl;




    });

    connect(editCameraButton, &QPushButton::clicked,[this](){
        this->cameraDialog->show();
    });

    connect(startButton, &QPushButton::pressed,[this](){
      //this->settingWidget->setDisabled(true);
      //this->render_policy_editor->setDisabled(true);
      //this->volumeRenderWidget->startRecording();
      this->startButton->setEnabled(false);
      //this->stopButton->setEnabled(true);
    });

//    connect(stopButton, &QPushButton::pressed,[this](){
//      this->settingWidget->setEnabled(true);
//
//      this->render_policy_editor->setEnabled(true);
//      this->volumeRenderWidget->stopRecording();
//      this->startButton->setEnabled(true);
//      this->stopButton->setEnabled(false);
//
//      std::vector<float> tf;
//      tf.resize(256);
//      this->settingWidget->getTransferFunc(tf.data());
//      this->volumeRenderWidget->setTransferFunction(tf.data(),256, true);
//
//      this->render_policy_editor->getRenderPolicy(renderPolicy.data());
//      this->volumeRenderWidget->setRenderPolicy(renderPolicy.data(),renderPolicy.size(), true);
//
//      this->volumeRenderWidget->setFPS(this->fps->value());
//      this->volumeRenderWidget->setIfSaveImage(this->imageSaving->isChecked());
//      this->volumeRenderWidget->setFileName(this->name->text().toStdString());
//
//      this->volumeRenderWidget->offScreenRender();
//    });


}

void OffScreenRenderSettingWidget::initCameraDialog()
{
    //camera dialog
    cameraDialog = new QDialog;
    cameraDialog->setWindowTitle("camera sequence");
    {
        auto cameraDialogLayout = new QHBoxLayout;

        cameraList = new QListWidget(this->cameraDialog);
        cameraDialogLayout->addWidget(cameraList);

        auto cameraDialogRightLayout = new QVBoxLayout;

        auto cameraDetailLayout = new QVBoxLayout;
        auto zoomLabel = new QLabel("zoom");
        cameraDetailLayout->addWidget(zoomLabel);
        zoomBox = new QDoubleSpinBox;
        cameraDetailLayout->addWidget(zoomBox);
        zoomBox->setDecimals(5);

        auto posLabel = new QLabel("pos");
        cameraDetailLayout->addWidget(posLabel);
        auto posLayout = new QHBoxLayout;
        auto posxLabel = new QLabel("x");
        posLayout->addWidget(posxLabel);
        posxBox = new QDoubleSpinBox;
        posLayout->addWidget(posxBox);
        auto posyLabel = new QLabel("y");
        posLayout->addWidget(posyLabel);
        posyBox = new QDoubleSpinBox;
        posLayout->addWidget(posyBox);
        auto poszLabel = new QLabel("z");
        posLayout->addWidget(poszLabel);
        poszBox = new QDoubleSpinBox;
        posLayout->addWidget(poszBox);
        cameraDetailLayout->addLayout(posLayout);
        posxBox->setDecimals(5);
        posyBox->setDecimals(5);
        poszBox->setDecimals(5);

        auto lookAtLabel = new QLabel("look at");
        cameraDetailLayout->addWidget(lookAtLabel);
        auto lookAtLayout = new QHBoxLayout;
        auto lookAtxLabel = new QLabel("x");
        lookAtLayout->addWidget(lookAtxLabel);
        lookAtxBox = new QDoubleSpinBox;
        lookAtLayout->addWidget(lookAtxBox);
        auto lookAtyLabel = new QLabel("y");
        lookAtLayout->addWidget(lookAtyLabel);
        lookAtyBox = new QDoubleSpinBox;
        lookAtLayout->addWidget(lookAtyBox);
        auto lookAtzLabel = new QLabel("z");
        lookAtLayout->addWidget(lookAtzLabel);
        lookAtzBox = new QDoubleSpinBox;
        lookAtLayout->addWidget(lookAtzBox);
        cameraDetailLayout->addLayout(lookAtLayout);
        lookAtxBox->setDecimals(5);
        lookAtyBox->setDecimals(5);
        lookAtzBox->setDecimals(5);

        auto upLabel = new QLabel("up");
        cameraDetailLayout->addWidget(lookAtLabel);
        auto upLayout = new QHBoxLayout;
        auto upxLabel = new QLabel("x");
        upLayout->addWidget(upxLabel);
        upxBox = new QDoubleSpinBox;
        upLayout->addWidget(upxBox);
        auto upyLabel = new QLabel("y");
        upLayout->addWidget(upyLabel);
        upyBox = new QDoubleSpinBox;
        upLayout->addWidget(upyBox);
        auto upzLabel = new QLabel("z");
        upLayout->addWidget(upzLabel);
        upzBox = new QDoubleSpinBox;
        upLayout->addWidget(upzBox);
        cameraDetailLayout->addLayout(upLayout);
        upxBox->setDecimals(5);
        upyBox->setDecimals(5);
        upzBox->setDecimals(5);

        cameraDialogRightLayout->addLayout(cameraDetailLayout);

        quitButton = new QPushButton("quit");
        cameraDialogRightLayout->addWidget(quitButton);

        cameraDialogLayout->addLayout(cameraDialogRightLayout);

        cameraDialog->setLayout(cameraDialogLayout);
    }

    connect(quitButton,&QPushButton::clicked,[this](){
      this->cameraDialog->close();
    });

    connect(cameraList,&QListWidget::itemSelectionChanged, [this](){
      auto text = cameraList->currentItem()->text();
      int idx=0;
      for(auto c:text){
          if(c >= '0' && c <= '9'){
              idx = idx * 10 + (c.toLatin1() - '0');
          }
      }

      Camera& camera = *this->cameraSequence[idx];
      zoomBox->setValue(camera.zoom);
      posxBox->setValue(camera.pos[0]);
      posyBox->setValue(camera.pos[1]);
      poszBox->setValue(camera.pos[2]);
      lookAtxBox->setValue(camera.look_at[0]);
      lookAtyBox->setValue(camera.look_at[1]);
      lookAtzBox->setValue(camera.look_at[2]);
      upxBox->setValue(camera.up[0]);
      upyBox->setValue(camera.up[1]);
      upzBox->setValue(camera.up[2]);
    });
}

void OffScreenRenderSettingWidget::volumeLoaded(std::string comp_volume_config)
{
    volume_data_config = comp_volume_config;

//    int iGPU = 0;
//    SetCUDACtx(iGPU);//used for cuda-renderer and volume
//    offScreenRenderer = CUDAOffScreenCompVolumeRenderer::Create(frameWidth, frameHeight);
    startButton->setEnabled(true);

//    space_x->setEnabled(false);
//    space_y->setEnabled(false);
//    space_z->setEnabled(false);

//    volumeForOffScreen = CompVolume::Load(m_comp_volume_path.c_str());
//    volumeForOffScreen->SetSpaceX(space[0]);
//    volumeForOffScreen->SetSpaceY(space[1]);
//    volumeForOffScreen->SetSpaceZ(space[2]);
//    offScreenRenderer->SetVolume(volumeForOffScreen);
//    offScreenRenderer->SetStep(0.00016,3000);

//    copyButton->setEnabled(true);

//    auto volume = volumeRenderWidget->getVolumeForRealTime();
//    float max_lod = std::max(volume->GetVolumeSpaceX() * volume->GetVolumeDimX(),
//                             volume->GetVolumeSpaceY() * volume->GetVolumeDimY());
//    max_lod = std::max(max_lod, volume->GetVolumeSpaceZ() * volume->GetVolumeDimZ());
//    render_policy_editor->init(max_lod);
}

void OffScreenRenderSettingWidget::volumeClosed(){
    spdlog::info("{0}.",__FUNCTION__ );
//    volumeForOffScreen.reset();
//    offScreenRenderer.reset();
    startButton->setEnabled(false);
//    stopButton->setEnabled(false);
//    copyButton->setEnabled(false);
 //   render_policy_editor->volumeClosed();
}

