//
// Created by csh on 10/20/2021.
//

#include "OffScreenRenderSetting.h"

static std::string GetName(const std::string& name){
    std::string s;
    s.reserve(name.size());
    for(auto c:name){
        if(c != '.'){
            s.push_back(c);
        }
        else{
            return s;
        }
    }
    return s;
}

static bool cameraEqual(const Camera& a,const Camera& b){
    if(a.zoom != b.zoom || a.n != b.n ||  a.f != b.f)
        return false;
    if(a.pos[0] != b.pos[0] || a.pos[1] != b.pos[1] || a.pos[2] != b.pos[2])
        return false;
    if(a.look_at[0] != b.look_at[0] || a.look_at[1] != b.look_at[1] || a.look_at[2] != b.look_at[2])
        return false;
    if(a.up[0] != b.up[0] || a.up[1] != b.up[1] || a.up[2] != b.up[2])
        return false;
    if(a.right[0] != b.right[0] || a.right[1] != b.right[1] || a.right[2] != b.right[2])
        return false;
    return true;
}

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

    initQualitySetting();

    auto widgetLayout = new QVBoxLayout;
    widgetLayout->setAlignment(Qt::AlignTop);
    auto settingLayout = new QVBoxLayout;
    settingLayout->setAlignment(Qt::AlignTop);

    auto volumeGroupBox = new QGroupBox("Volume");
    volumeGroupBox->setFixedHeight(50);
    auto volumeLayout = new QVBoxLayout;
    volumeGroupBox->setLayout(volumeLayout);
    volumeNameLabel = new QLabel("volume not loaded");
    volumeNameLabel->setWordWrap(true);
    volumeLayout->addWidget(volumeNameLabel);
    settingLayout->addWidget(volumeGroupBox);

    auto renderQualityGroupBox = new QGroupBox("render quality");
    renderQualityGroupBox->setFixedHeight(80);
    auto renderQualityLayout = new QHBoxLayout;
    renderQualityGroupBox->setLayout(renderQualityLayout);
    auto lowLabel = new QLabel("low");
    lowLabel->setFixedWidth(70);
    auto highLabel = new QLabel("high");
    highLabel->setFixedWidth(70);
    qualitySlider = new QSlider(Qt::Horizontal);
    qualitySlider->setRange(0,3);
    renderQualityLayout->addWidget(lowLabel);
    renderQualityLayout->addWidget(qualitySlider);
    renderQualityLayout->addWidget(highLabel);
    settingLayout->addWidget(renderQualityGroupBox);

    auto cameraGroupBox = new QGroupBox("Camera sequence");
    cameraGroupBox->setFixedHeight(150);
    auto cameraGroupBoxLayout = new QVBoxLayout;
    cameraGroupBox->setLayout(cameraGroupBoxLayout);
    loadCameraButton = new QPushButton("load camera sequence");
    smoothCameraButton = new QPushButton("smooth camera route");
//    editCameraButton = new QPushButton("edit camera sequence");
    cameraFileLabel =new QLabel("Camera sequence file not loaded.");
    cameraFileLabel->setWordWrap(true);
    cameraGroupBoxLayout->addWidget(cameraFileLabel);
    cameraGroupBoxLayout->addWidget(loadCameraButton);
    cameraGroupBoxLayout->addWidget(smoothCameraButton);
//    cameraGroupBoxLayout->addWidget(editCameraButton);
    settingLayout->addWidget(cameraGroupBox);
//    editCameraButton->setVisible(false);

    auto tfGroupBox = new QGroupBox("Transfer function");
    tfGroupBox->setFixedHeight(120);
    auto tfGroupBoxLayout = new QVBoxLayout;
    tfGroupBox->setLayout(tfGroupBoxLayout);
    tfLabel =  new QLabel("Transfer function file not loaded. Please edit on the left.");
    tfLabel->setWordWrap(true);
    tfGroupBoxLayout->addWidget(tfLabel);
    loadTFButton = new QPushButton("load transfer function file");
    tfGroupBoxLayout->addWidget(loadTFButton);
    settingLayout->addWidget(tfGroupBox);

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
    settingLayout->addWidget(videoSettingGroupBox);

    widgetLayout->addLayout(settingLayout);

    auto buttonLayout = new QHBoxLayout;
    loadSettingButton = new QPushButton("load setting file");
    buttonLayout->addWidget(loadSettingButton);
    saveButton = new QPushButton("save setting file");
    saveButton->setEnabled(false);
    buttonLayout->addWidget(saveButton);
    renderButton = new QPushButton("render");
    renderButton->setEnabled(false);
    buttonLayout->addWidget(renderButton);
    widgetLayout->addLayout(buttonLayout);

    this->setLayout(widgetLayout);

//    initCameraDialog();

    connect(smoothCameraButton,&QPushButton::clicked,this,&OffScreenRenderSettingWidget::smoothCameraRoute);

    connect(loadSettingButton,&QPushButton::clicked,[this](){
       this->settingFile =  QFileDialog::getOpenFileName(this,
                                                         QStringLiteral("OpenFile"),
                                                         QStringLiteral("."),
                                                         QStringLiteral("config files(*.json)")
       ).toStdString();
       this->loadSettingFile();
    });

    connect(loadTFButton,&QPushButton::clicked,[this](){
      this->tfFile = QFileDialog::getOpenFileName(this,
                                                  QStringLiteral("OpenFile"),
                                                  QStringLiteral("."),
                                                  QStringLiteral("config files(*.json)")
      ).toStdString();
      this->loadTransferFuncFile();
    });

    connect(loadCameraButton,&QPushButton::clicked,[this](){
      this->camera_sequence_config = QFileDialog::getOpenFileName(this,
                                   QStringLiteral("OpenFile"),
                                   QStringLiteral("."),
                                   QStringLiteral("config files(*.json)")
      ).toStdString();
      this->loadCameraSequenceFile();
    });

//    connect(editCameraButton, &QPushButton::clicked,[this](){
//        this->cameraDialog->show();
//    });

    connect(saveButton,&QPushButton::clicked,this,&OffScreenRenderSettingWidget::saveSettingFile);
    connect(renderButton, &QPushButton::pressed,this,&OffScreenRenderSettingWidget::render);
    connect(name,&QLineEdit::textChanged,[this](){
        output_video_name = name->text().toStdString() + ".avi";
    });
}

void OffScreenRenderSettingWidget::loadSettingFile()
{
    std::ifstream in;
    in.open(settingFile);
    if(!in.is_open()){
        spdlog::error("Open setting file failed");
        throw std::runtime_error("Open setting file failed");
    }

    nlohmann::json j;
    in >> j;

    if(j.find("volume_data_config")!=j.end()){
        volume_data_config = j.at("volume_data_config");
    }
    else{
        volume_data_config = "C:/Users/csh/project/VolumeSlicer/bin/mouse_file_config.json";
        spdlog::error("volume_data_config file not provided, use C:/Users/csh/project/VolumeSlicer/bin/mouse_file_config.json");
    }
    volumeNameLabel->setText(QString::fromStdString(volume_data_config));

    if(j.find("render_quality_level")!=j.end()){
        qualitySlider->setValue(j.at("render_quality_level"));
    }
    else{
        spdlog::error("render quality level not provided.");
    }

    if(j.find("save_image")!=j.end()){
        if(j.at("save_image")=="yes")
            imageSaving->setChecked(true);
        else{
            imageSaving->setChecked(false);
        }
    }
    else{
        spdlog::error("save_image not provided.");
    }

    if(j.find("output_video_name")!=j.end()){
        output_video_name = j.at("output_video_name");
        name->setText(QString::fromStdString(GetName(output_video_name)));
    }
    else{
        spdlog::error("output video name not provided.");
    }

    if(j.find("camera_sequence_config")!=j.end()){
        camera_sequence_config = j.at("camera_sequence_config");
        loadCameraSequenceFile();
    }
    else{
        fps = 30;
        spdlog::error("camera sequence config not provided.");
    }

    if(j.find("transfer_func_config")!=j.end()){
        tfFile = j.at("transfer_func_config");
        loadTransferFuncFile();
    }
    else{
        spdlog::error("transfer func config not provided.");
    }
}

void OffScreenRenderSettingWidget::loadTransferFuncFile()
{
    std::ifstream in;
    in.open(tfFile);
    if(!in.is_open()){
        spdlog::error("Open transfer func file failed");
        throw std::runtime_error("Open transfer func file failed");
    }

    nlohmann::json j;
    in >> j;

    for(int i=0;i<256;i++){
        std::string idx = std::to_string(i);
        if(j.find(idx) != j.end()){
            tf.push_back(j[idx][0]);
            tf.push_back(j[idx][1]);
            tf.push_back(j[idx][2]);
            tf.push_back(j[idx][3]);
        }
        else{
            tf.push_back(0);
            tf.push_back(0);
            tf.push_back(0);
            tf.push_back(0);
            spdlog::error("Not provide {}, use rgba(0,0,0,0)",idx);
        }
    }

    tfLabel->setText(("Transfer function file:"+tfFile).c_str());
}

void OffScreenRenderSettingWidget::render(){
    if(cameraSequence.empty()){
        auto errorDialog = new QDialog;
        errorDialog->setWindowTitle("error");
        auto errorLabel = new QLabel("Camera sequence not loaded.");
        auto errorLayout = new QVBoxLayout;
        errorLayout->addWidget(errorLabel);
        errorDialog->setLayout(errorLayout);
        errorDialog->show();
        return;
    }

    int iGPU = 0;
    SetCUDACtx(iGPU);//used for cuda-renderer and volume
    offScreenRenderer = CUDAOffScreenCompVolumeRenderer::Create(frameWidth, frameHeight);

    volumeForOffScreen = CompVolume::Load(volume_data_config.c_str());
    volumeForOffScreen->SetSpaceX(space[0]);
    volumeForOffScreen->SetSpaceY(space[1]);
    volumeForOffScreen->SetSpaceZ(space[2]);
    offScreenRenderer->SetVolume(volumeForOffScreen);
    offScreenRenderer->SetStep(0.00016,3000);

    //lod policy
    int max_lod = sizeof(CompRenderPolicy::lod_dist) / sizeof(CompRenderPolicy::lod_dist[0]);
    CompRenderPolicy policy;
    int idx = qualitySlider->value();
    for (int i = 0; i < renderPolicy[idx].size() && i < max_lod; i++)
    {
        policy.lod_dist[i] = renderPolicy[idx][i];
    }
    policy.cdf_value_file="chebyshev_dist_mouse_cdf_config.json";
    offScreenRenderer->SetRenderPolicy(policy);
    std::cout<<"after set rp"<<std::endl;

    //transfer func
    if(tfFile.empty()){
        tf.clear();
        tf.resize(256*4);
        settingWidget->getTransferFunc(tf.data());
    }

    TransferFunc transferFunc;
    for(int i = 0;i < 256;i++){
        transferFunc.points.emplace_back(i,std::array<double,4>{tf[i * 4 + 0],tf[i * 4 + 1],tf[i * 4 + 2],tf[i * 4 + 3]});
    }
    offScreenRenderer->SetTransferFunc(std::move(transferFunc));
    std::cout<<"after set tf"<<std::endl;

    VideoCapture video_capture((name->text().toStdString() + ".avi").c_str(),frameWidth,frameHeight,fps);

    //render
    int save_image = imageSaving->isChecked();
    std::cout<<"sequence size "<<cameraSequence.size()<<std::endl;
    for(int i = 0 ;i < cameraSequence.size(); i++){
        std::cout<<"render "<<i<<std::endl;

        if(i > 0 && cameraEqual(*(cameraSequence[i]),*(cameraSequence[i - 1])))
        {
            video_capture.AddLastFrame();
            continue;
        }
        Timer mTimer;
        mTimer.start();
        offScreenRenderer->SetCamera(*cameraSequence[i]);
        std::cout<<"after set camera "<<std::endl;
        offScreenRenderer->render(true);
        std::cout<<"after render"<<std::endl;
        auto image = offScreenRenderer->GetImage();
        std::cout<<"get image "<<i<<std::endl;
        if(save_image){
            if(!std::filesystem::exists("images")){
                std::filesystem::create_directory("images");
            }
            image.SaveToFile(("images/"+name->text().toStdString()+"_frame_"+std::to_string(i)+".jpeg").c_str());
        }
        std::cout<<"before"<<" "<<i<<std::endl;
        auto img = image.ToImage3b();
        video_capture.AddFrame(reinterpret_cast<uint8_t*>(img.GetData()));
        std::cout<<"after"<<std::endl;
        mTimer.stop();
//        spdlog::set_level(spdlog::level::info);
//        LogInfo("render frame "+std::to_string(i)+" cost time "+mTimer.duration().s().fmt());
        //for gpu take a rest
        Sleep(2000);
    }

    offScreenRenderer.reset();
    volumeForOffScreen.reset();
}

void OffScreenRenderSettingWidget::loadCameraSequenceFile(){
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
    if(j.find("fps") != j.end()){
        fps = j.at("fps");
    }
    else{
        fps = 30;
        spdlog::error("Not provide fps, use default fps(30)");
    }

    saveButton->setEnabled(true);
    renderButton->setEnabled(true);

//    this->cameraList->clear();
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

//        this->cameraList->addItem(frame_idx.c_str());

        this->cameraSequence.emplace_back(std::move(camera));
    }

    cameraFileLabel->setText(("Camera sequence file:"+camera_sequence_config).c_str());
}

void OffScreenRenderSettingWidget::saveSettingFile()
{
    auto time = std::time(0);
    auto tm = localtime(&time);
    std::string tmStr = std::to_string(tm->tm_mon)+"-"+std::to_string(tm->tm_mday)+" "+std::to_string(tm->tm_hour)+"h"+std::to_string(tm->tm_min)+"m"+std::to_string(tm->tm_sec)+"s";
    std::string fileName = tmStr + " offscreen_render_config.json";

    nlohmann::json j;
    j["volume_data_config"]=volume_data_config;
    j["render_quality_level"] = qualitySlider->value();
    j["camera_sequence_config"] = camera_sequence_config;
    if(imageSaving->isChecked())
        j["save_image"]="yes";
    else
        j["save_image"]="no";
    j["output_video_name"] = (name->text().toStdString()+".avi").c_str();
    if(!tfFile.empty())
        j["transfer_func_config"]=tfFile;
    else{
        j["transfer_func_config"]=settingWidget->saveTFFile(name->text().toStdString());
    }

    std::ofstream out(fileName);
    out << j <<std::endl;
    out.close();
}

void OffScreenRenderSettingWidget::initQualitySetting()
{
    float max = std::numeric_limits<float>::max();
    float data1[] = {0.1,0.8,2.0,3.2,5.0,8.6,max};
    renderPolicy.emplace_back(std::vector<float>(data1,data1+7));

    float data2[] = {0.3,1.0,2.6,4.8,6.4,9.0,max};
    renderPolicy.emplace_back(std::vector<float>(data2,data2+7));

    float data3[] = {0.6,1.4,3.0,6.2,7.8,9.4,max};
    renderPolicy.emplace_back(std::vector<float>(data3,data3+7));

    float data4[] = {0.8,1.6,3.2,6.4,8.0,9.6,max};
    renderPolicy.emplace_back(std::vector<float>(data4,data4+7));

//    float data5[] = {0.8,1.6,3.2,6.4,8.0,9.6,-1.0};
//    renderPolicy.emplace_back(std::vector<float>(data5,data5+7));
}

//void OffScreenRenderSettingWidget::initCameraDialog()
//{
//    //camera dialog
//    cameraDialog = new QDialog;
//    cameraDialog->setWindowTitle("camera sequence");
//    {
//        auto cameraDialogLayout = new QHBoxLayout;
//
//        cameraList = new QListWidget(this->cameraDialog);
//        cameraDialogLayout->addWidget(cameraList);
//
//        auto cameraDialogRightLayout = new QVBoxLayout;
//
//        auto cameraDetailLayout = new QVBoxLayout;
//        auto zoomLabel = new QLabel("zoom");
//        cameraDetailLayout->addWidget(zoomLabel);
//        zoomBox = new QDoubleSpinBox;
//        cameraDetailLayout->addWidget(zoomBox);
//        zoomBox->setDecimals(5);
//
//        auto posLabel = new QLabel("pos");
//        cameraDetailLayout->addWidget(posLabel);
//        auto posLayout = new QHBoxLayout;
//        auto posxLabel = new QLabel("x");
//        posLayout->addWidget(posxLabel);
//        posxBox = new QDoubleSpinBox;
//        posLayout->addWidget(posxBox);
//        auto posyLabel = new QLabel("y");
//        posLayout->addWidget(posyLabel);
//        posyBox = new QDoubleSpinBox;
//        posLayout->addWidget(posyBox);
//        auto poszLabel = new QLabel("z");
//        posLayout->addWidget(poszLabel);
//        poszBox = new QDoubleSpinBox;
//        posLayout->addWidget(poszBox);
//        cameraDetailLayout->addLayout(posLayout);
//        posxBox->setDecimals(5);
//        posyBox->setDecimals(5);
//        poszBox->setDecimals(5);
//
//        auto lookAtLabel = new QLabel("look at");
//        cameraDetailLayout->addWidget(lookAtLabel);
//        auto lookAtLayout = new QHBoxLayout;
//        auto lookAtxLabel = new QLabel("x");
//        lookAtLayout->addWidget(lookAtxLabel);
//        lookAtxBox = new QDoubleSpinBox;
//        lookAtLayout->addWidget(lookAtxBox);
//        auto lookAtyLabel = new QLabel("y");
//        lookAtLayout->addWidget(lookAtyLabel);
//        lookAtyBox = new QDoubleSpinBox;
//        lookAtLayout->addWidget(lookAtyBox);
//        auto lookAtzLabel = new QLabel("z");
//        lookAtLayout->addWidget(lookAtzLabel);
//        lookAtzBox = new QDoubleSpinBox;
//        lookAtLayout->addWidget(lookAtzBox);
//        cameraDetailLayout->addLayout(lookAtLayout);
//        lookAtxBox->setDecimals(5);
//        lookAtyBox->setDecimals(5);
//        lookAtzBox->setDecimals(5);
//
//        auto upLabel = new QLabel("up");
//        cameraDetailLayout->addWidget(upLabel);
//        auto upLayout = new QHBoxLayout;
//        auto upxLabel = new QLabel("x");
//        upLayout->addWidget(upxLabel);
//        upxBox = new QDoubleSpinBox;
//        upLayout->addWidget(upxBox);
//        auto upyLabel = new QLabel("y");
//        upLayout->addWidget(upyLabel);
//        upyBox = new QDoubleSpinBox;
//        upLayout->addWidget(upyBox);
//        auto upzLabel = new QLabel("z");
//        upLayout->addWidget(upzLabel);
//        upzBox = new QDoubleSpinBox;
//        upLayout->addWidget(upzBox);
//        cameraDetailLayout->addLayout(upLayout);
//        upxBox->setDecimals(5);
//        upyBox->setDecimals(5);
//        upzBox->setDecimals(5);
//
//        cameraDialogRightLayout->addLayout(cameraDetailLayout);
//
//        quitButton = new QPushButton("quit");
//        cameraDialogRightLayout->addWidget(quitButton);
//
//        cameraDialogLayout->addLayout(cameraDialogRightLayout);
//
//        cameraDialog->setLayout(cameraDialogLayout);
//    }
//
//    connect(quitButton,&QPushButton::clicked,[this](){
//      this->cameraDialog->close();
//    });
//
//    connect(cameraList,&QListWidget::itemSelectionChanged, [this](){
//      auto text = cameraList->currentItem()->text();
//      int idx=0;
//      for(auto c:text){
//          if(c >= '0' && c <= '9'){
//              idx = idx * 10 + (c.toLatin1() - '0');
//          }
//      }
//
//      Camera& camera = *this->cameraSequence[idx];
//      zoomBox->setValue(camera.zoom);
//      posxBox->setValue(camera.pos[0]);
//      posyBox->setValue(camera.pos[1]);
//      poszBox->setValue(camera.pos[2]);
//      lookAtxBox->setValue(camera.look_at[0]);
//      lookAtyBox->setValue(camera.look_at[1]);
//      lookAtzBox->setValue(camera.look_at[2]);
//      upxBox->setValue(camera.up[0]);
//      upyBox->setValue(camera.up[1]);
//      upzBox->setValue(camera.up[2]);
//    });
//}

void OffScreenRenderSettingWidget::volumeLoaded(std::string comp_volume_config)
{
    volume_data_config = comp_volume_config;
    volumeNameLabel->setText(volume_data_config.c_str());
//    int iGPU = 0;
//    SetCUDACtx(iGPU);//used for cuda-renderer and volume
//    offScreenRenderer = CUDAOffScreenCompVolumeRenderer::Create(frameWidth, frameHeight);

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
//    saveButton->setEnabled(false);
//    renderButton->setEnabled(false);
}

void OffScreenRenderSettingWidget::smoothCameraRoute()
{

}