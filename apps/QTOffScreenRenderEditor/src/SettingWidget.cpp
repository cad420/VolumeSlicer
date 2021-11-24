//
// Created by csh on 10/21/2021.
//

#include "SettingWidget.h"

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

SettingWidget::SettingWidget(VolumeRenderWidget* volumeRenderWidget, QWidget* parent):volumeRenderWidget(volumeRenderWidget),
    QWidget(parent)
{
    auto widgetLayout = new QVBoxLayout;
    widgetLayout->setAlignment(Qt::AlignTop);

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
    widgetLayout->addWidget(renderQualityGroupBox);
    initQualitySetting();

    auto cameraGroupBox = new QGroupBox("Camera sequence");
    cameraGroupBox->setFixedHeight(150);
    auto cameraGroupBoxLayout = new QVBoxLayout;
    cameraGroupBox->setLayout(cameraGroupBoxLayout);
    loadCameraButton = new QPushButton("load");
    optimizeCameraButton = new QPushButton("optimize");
//    editCameraButton = new QPushButton("edit camera sequence");
    cameraGroupBoxLayout->addWidget(loadCameraButton);
    cameraGroupBoxLayout->addWidget(optimizeCameraButton);
//    cameraGroupBoxLayout->addWidget(editCameraButton);
    widgetLayout->addWidget(cameraGroupBox);

    auto tfGroupBox = new QGroupBox("Transfer function");
    tfGroupBox->setFixedHeight(500);
    auto tfLayout =  new QVBoxLayout;
    tfGroupBox->setLayout(tfLayout);
    tf_editor = new TF1DEditor(this);
    tf_editor->setFixedHeight(400);
    tf.resize(256*4,0.f);
    tfLayout->addWidget(tf_editor);
    widgetLayout->addWidget(tfGroupBox);
    tf_editor->setEnabled(false);

    auto videoSettingLayout = new QVBoxLayout;
    auto videoSettingGroupBox = new QGroupBox("Video Setting");
    videoSettingGroupBox->setLayout(videoSettingLayout);
    videoSettingGroupBox->setFixedHeight(100);

    auto fpsLayout = new QHBoxLayout;
    auto fpsLabel = new QLabel("fps");
    fpsLabel->setAlignment(Qt::AlignCenter);
    fpsSpinBox = new QSpinBox;
    fpsSpinBox->setRange(10,70);
    fpsSpinBox->setValue(30);
    fpsLayout->addWidget(fpsLabel);
    fpsLayout->addWidget(fpsSpinBox);
    videoSettingLayout->addLayout(fpsLayout);

    auto checkBoxLayout = new QHBoxLayout;
    auto checkBoxLabel = new QLabel("save image");
    checkBoxLabel->setAlignment(Qt::AlignHCenter);
    imageSaving = new QCheckBox;
    checkBoxLayout->addWidget(checkBoxLabel);
    checkBoxLayout->addWidget(imageSaving);
    videoSettingLayout->addLayout(checkBoxLayout);

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

    auto settingButtonLayout = new QHBoxLayout;
    loadSettingButton = new QPushButton("load setting file");
    settingButtonLayout->addWidget(loadSettingButton);
    saveSettingButton = new QPushButton("save setting file");
//    saveSettingButton->setEnabled(false);
    settingButtonLayout->addWidget(saveSettingButton);
    widgetLayout->addLayout(settingButtonLayout);

    auto buttonsLayout = new QHBoxLayout;
    startButton = new QPushButton("start recording");
    startButton->setEnabled(true);
    stopButton = new QPushButton("stop recording");
    stopButton->setEnabled(false);
    buttonsLayout->addWidget(startButton);
    buttonsLayout->addWidget(stopButton);
    widgetLayout->addLayout(buttonsLayout);

    this->setLayout(widgetLayout);

    connect(fpsSpinBox,&QSpinBox::valueChanged,[this](){
        this->volumeRenderWidget->setFPS(this->fpsSpinBox->value());
    });

    connect(startButton, &QPushButton::pressed,[this](){
       this->volumeRenderWidget->startRecording();
       this->tf_editor->setEnabled(false);
       this->startButton->setEnabled(false);
       this->stopButton->setEnabled(true);
    });

    connect(stopButton, &QPushButton::pressed,[this](){
      this->volumeRenderWidget->stopRecording();
      this->tf_editor->setEnabled(true);
      this->startButton->setEnabled(true);
      this->stopButton->setEnabled(false);
    });

    connect(tf_editor,&TF1DEditor::TF1DChanged,[this](){
        this->tf_editor->getTransferFunction(tf.data(),256,1.0);
        this->volumeRenderWidget->setTransferFunction(tf.data(),256);

//        tf.clear();
//        index.clear();
//        tf.resize(256*4,0.f);
//        index.resize(256,0.f);
//        int num[1];
//        this->tf_editor->getTransferFunction(tf.data(),index.data(),num, 256,1.0);
//        this->volumeRenderWidget->setTransferFunction(tf.data(),index.data(), *num);
    });

    connect(loadCameraButton,&QPushButton::clicked,[this](){
      this->camera_sequence_config = QFileDialog::getOpenFileName(this,
                                                                  QStringLiteral("OpenFile"),
                                                                  QStringLiteral("."),
                                                                  QStringLiteral("config files(*.json)")
      ).toStdString();
      this->loadCameraSequenceFile();
    });

    connect(saveSettingButton,&QPushButton::clicked,this,&SettingWidget::saveSettingFile);

    connect(loadSettingButton,&QPushButton::clicked,[this](){
      this->setting_file = QFileDialog::getOpenFileName(this,
                                                                  QStringLiteral("OpenFile"),
                                                                  QStringLiteral("."),
                                                                  QStringLiteral("config files(*.json)")
      ).toStdString();
      this->loadSettingFile();
    });
}

void SettingWidget::saveSettingFile()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save TF1D"), QString::fromStdString(setting_path), tr("Camera Sequence(*.json);;All Files (*)"));
    setting_path = QFileInfo(fileName).absolutePath().toStdString();
//    auto time = std::time(0);
//    auto tm = localtime(&time);
//    std::string tmStr = std::to_string(tm->tm_mon)+"-"+std::to_string(tm->tm_mday)+" "+std::to_string(tm->tm_hour)+"h"+std::to_string(tm->tm_min)+"m"+std::to_string(tm->tm_sec)+"s";
//    std::string fileName = tmStr + " offscreen_render_config.json";

    nlohmann::json j;
    j["volume_file"] = volume_file;
    j["fps"] = fpsSpinBox->value();
    j["backend"] = "cuda";
    j["iGPU"] = 0;
    j["width"] = 700;
    j["height"] = 700;
    j["output_video_name"] = name->text().toStdString() + ".avi";
    j["save_image"] = imageSaving->isChecked()?"yes":"no";
    j["volume_data_config"]=comp_config_path;
    j["space"] = {0.00032,0.00032,0.001};


    int idx = qualitySlider->value();
    j["lod_policy"] ={renderPolicy[idx][0],renderPolicy[idx][1],renderPolicy[idx][2],renderPolicy[idx][3],renderPolicy[idx][4],renderPolicy[idx][5],renderPolicy[idx][6]};
    j["render_quality_level"] = idx;

    //camera sequence
    if(!camera_sequence_config.empty())
        j["camera_sequence_config"]=camera_sequence_config;
    else{
        QMessageBox::critical(this, tr("Error"),
                              tr("camera sequence config file not provided."));
        return;
    }

    //tf
    tf_editor->getTransferFunction(tf.data(),256,1.0);
    for(int i=0;i<255;i++)
        j["tf"][std::to_string(i)]={tf[i*4],tf[i*4+1],tf[i*4+2],tf[i*4+3]};
    tfFile = name->text().toStdString() + "_transfer_function";
    tf_editor->saveTransferFunctionWithTitle(tfFile);
    j["TF1D"]= QApplication::applicationDirPath().toStdString() + '/'+ tfFile+".TF1D";

    std::ofstream out(fileName.toStdString());
    out << j <<std::endl;
    out.close();
}

void SettingWidget::loadSettingFile(){
    std::ifstream in;
    in.open(setting_file);
    if(!in.is_open()){
        spdlog::error("Open setting file failed");
        throw std::runtime_error("Open setting file failed");
    }

    nlohmann::json j;
    in >> j;

    if(j.find("volume_file")!=j.end()){
        volume_file=j.at("volume_file");
    }
    else{
        spdlog::error("volume_file not provided.");
    }
    volumeLoaded(volume_file);

    if(j.find("fps")!=j.end()){
        fpsSpinBox->setValue(j.at("fps"));
    }
    else{
        spdlog::error("fps not provided.");
    }

    if(j.find("volume_data_config")!=j.end()){
        comp_config_path = j.at("volume_data_config");
    }
    else{
        comp_config_path = "C:/Users/csh/project/VolumeSlicer/bin/mouse_file_config.json";
        spdlog::error("volume_data_config file not provided, use C:/Users/csh/project/VolumeSlicer/bin/mouse_file_config.json");
    }

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

    if(j.find("output_video_name")!=j.end())
    {
        //        output_video_name = j.at("output_video_name");
        name->setText(QString::fromStdString(GetName(j.at("output_video_name"))));
    }
    else{
        spdlog::error("output video name not provided.");
    }

    if(j.find("camera_sequence_config")!=j.end()){
        camera_sequence_config = j.at("camera_sequence_config");
        loadCameraSequenceFile();
    }
    else{
        spdlog::error("camera sequence config not provided.");
    }

    if(j.find("TF1D")!=j.end()){
        tfFile = j.at("TF1D");
        tf_editor->loadTransferFunction(QString::fromStdString(tfFile));
//        loadTransferFuncFile();
    }
    else{
        spdlog::error("transfer func config not provided.");
    }
}

void SettingWidget::loadCameraSequenceFile(){
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
}

void SettingWidget::getTransferFunc(float* tfData){
    this->tf_editor->getTransferFunction(tfData,256,1.0);
}

void SettingWidget::initQualitySetting()
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

std::string SettingWidget::saveTFFile(){
    auto time = std::time(0);
    auto tm = localtime(&time);
    std::string tmStr = std::to_string(tm->tm_mon)+"-"+std::to_string(tm->tm_mday)+" "+std::to_string(tm->tm_hour)+"h"+std::to_string(tm->tm_min)+"m"+std::to_string(tm->tm_sec)+"s";
    std::string fileName = tmStr + " transfer_function.json";

    nlohmann::json j;
    for(int i=0;i < 256;i++){
        std::string idx=std::to_string(i);
        j[idx]={
            tf[i*4+0],
            tf[i*4+1],
            tf[i*4+2],
            tf[i*4+3]
        };
    }

    std::ofstream out(fileName);
    out << j <<std::endl;
    out.close();

    return fileName;
}

void SettingWidget::setTF(float* tfData, int num){
    tf_editor->setTF(tfData,num);
}

void SettingWidget::volumeLoaded(std::string file, std::string comp_volume_config,std::string raw_volume_path,uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float space_x,float space_y, float space_z)
{

    comp_config_path=comp_volume_config;
    volume_file=file;

    volumeRenderWidget->setRenderPolicy(renderPolicy[0].data(), renderPolicy[0].size());

    rawVolume=RawVolume::Load(raw_volume_path.c_str(),VoxelType::UInt8, {dim_x, dim_y, dim_z}, {space_x, space_y, space_z});

    trivalVolume = std::make_unique<TrivalVolume>(rawVolume->GetData(),rawVolume->GetVolumeDimX(),
                                                rawVolume->GetVolumeDimY(),rawVolume->GetVolumeDimZ());
    tf_editor->setVolumeInformation(trivalVolume.get());
    tf_editor->setFixedHeight(400);
    tf.resize(256*4,0.f);
    tf_editor->setEnabled(true);
    tf_editor->getTransferFunction(tf.data(),256,1.0);
    volumeRenderWidget->setTransferFunction(tf.data(),256);

//    tf_editor->getTransferFunction(tf.data(),index.data(),256,1.0);
//    volumeRenderWidget->setTransferFunction(tf,index);

}

void SettingWidget::volumeLoaded(std::string filename){
    std::ifstream in(filename);
    volume_file=filename;

    if(!in.is_open()){
        QMessageBox::critical(NULL,"Error","File open failed!",QMessageBox::Yes);
        return;
    }

    try{
        nlohmann::json j;
        in>>j;

        auto comp_volume_info=j["comp_volume"];
        comp_config_path=comp_volume_info.at("comp_config_file_path");
        volumeRenderWidget->loadVolume(comp_config_path);
        volumeRenderWidget->setRenderPolicy(renderPolicy[0].data(), renderPolicy[0].size());
//        offscreen_render_setting_widget->volumeLoaded(comp_volume_path);

        auto raw_volume_info=j["raw_volume"];
        std::string raw_volume_path=raw_volume_info.at("raw_volume_path");
        auto raw_volume_dim=raw_volume_info.at("raw_volume_dim");
        auto raw_volume_space=raw_volume_info.at("raw_volume_space");
        uint32_t dim_x=raw_volume_dim.at(0);
        uint32_t dim_y=raw_volume_dim.at(1);
        uint32_t dim_z=raw_volume_dim.at(2);
        float space_x=raw_volume_space.at(0);
        float space_y=raw_volume_space.at(1);
        float space_z=raw_volume_space.at(2);
        rawVolume=RawVolume::Load(raw_volume_path.c_str(),VoxelType::UInt8, {dim_x, dim_y, dim_z}, {space_x, space_y, space_z});

        trivalVolume = std::make_unique<TrivalVolume>(rawVolume->GetData(),rawVolume->GetVolumeDimX(),
                                                      rawVolume->GetVolumeDimY(),rawVolume->GetVolumeDimZ());
        tf_editor->setVolumeInformation(trivalVolume.get());
        tf_editor->setFixedHeight(400);
        tf.resize(256*4,0.f);
        tf_editor->setEnabled(true);
        tf_editor->getTransferFunction(tf.data(),256,1.0);
        volumeRenderWidget->setTransferFunction(tf.data(),256);
    }
    catch (const std::exception& err) {
        QMessageBox::critical(NULL,"Error","Config file format error!",QMessageBox::Yes);
    }
}

void SettingWidget::volumeClosed(){
    spdlog::info("{0}.",__FUNCTION__ );
//    space_x->setEnabled(true);
//    space_y->setEnabled(true);
//    space_z->setEnabled(true);
    volumeFileLabel = new QLabel("volume not loaded");
    trivalVolume.reset();
    rawVolume.reset();
    tf_editor->resetTransferFunction();
    tf_editor->setEnabled(false);
    tf.clear();
    renderPolicy.clear();
//    render_policy_editor->volumeClosed();
}

void SettingWidget::setCameraName(std::string filename){
    camera_sequence_config=filename;
}

