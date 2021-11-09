//
// Created by csh on 10/21/2021.
//

#include "SettingWidget.h"

SettingWidget::SettingWidget(VolumeRenderWidget* volumeRenderWidget, QWidget* parent):volumeRenderWidget(volumeRenderWidget),
    QWidget(parent)
{
    renderPolicy.resize(10,0.f);
    renderPolicy[0]=0.8;
    renderPolicy[1]=1.6;
    renderPolicy[2]=3.3;
    renderPolicy[3]=std::numeric_limits<double>::max();

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

//    render_policy_editor = new RenderPolicyEditor(this);
//    render_policy_editor->setFixedHeight(150);
    renderPolicy.resize(10);
//    widgetLayout->addWidget(render_policy_editor);

    auto tfGroupBox = new QGroupBox("Transfer function");
    auto tfLayout =  new QHBoxLayout;
    tfGroupBox->setLayout(tfLayout);
    tf_editor = new TF1DEditor(this);
    tf_editor->setFixedHeight(400);
    tf.resize(256*4,0.f);
    //index.resize(256,0.f);
    tfLayout->addWidget(tf_editor);
    widgetLayout->addWidget(tfGroupBox);
    tf_editor->setEnabled(false);

    auto fpsLayout = new QHBoxLayout;
    auto fpsLabel = new QLabel("fps");
    fpsLabel->setAlignment(Qt::AlignCenter);
    fpsSpinBox = new QSpinBox;
    fpsSpinBox->setRange(10,70);
    fpsSpinBox->setValue(30);
    fpsLayout->addWidget(fpsLabel);
    fpsLayout->addWidget(fpsSpinBox);
    widgetLayout->addLayout(fpsLayout);

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
       //this->render_policy_editor->setEnabled(false);
       this->tf_editor->setEnabled(false);
       this->startButton->setEnabled(false);
       this->stopButton->setEnabled(true);
    });

    connect(stopButton, &QPushButton::pressed,[this](){
      this->volumeRenderWidget->stopRecording();
      //this->render_policy_editor->setEnabled(true);
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

//    connect(render_policy_editor,&RenderPolicyEditor::renderPolicyChanged,[this](){
//        this->render_policy_editor->getRenderPolicy(renderPolicy.data());
//        this->volumeRenderWidget->setRenderPolicy(renderPolicy.data(), renderPolicy.size());
//    });
}

void SettingWidget::getTransferFunc(float* tfData){
    this->tf_editor->getTransferFunction(tfData,256,1.0);
}

void SettingWidget::setTF(float* tfData, int num){
    tf_editor->setTF(tfData,num);
}

void SettingWidget::volumeLoaded(std::string raw_volume_path,uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float space_x,float space_y, float space_z)
{
//    auto volume = volumeRenderWidget->getVolumeForRealTime();
//    space_x->setEnabled(false);
//    space_y->setEnabled(false);
//    space_z->setEnabled(false);

//    float max_lod = std::max(volume->GetVolumeSpaceX() * volume->GetVolumeDimX(),
//                              volume->GetVolumeSpaceY() * volume->GetVolumeDimY());
//    max_lod = std::max(max_lod, volume->GetVolumeSpaceZ() * volume->GetVolumeDimZ());
//    render_policy_editor->init(max_lod);
//    render_policy_editor->getRenderPolicy(renderPolicy.data());
    volumeRenderWidget->setRenderPolicy(renderPolicy.data(), renderPolicy.size());

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

void SettingWidget::volumeClosed(){
    spdlog::info("{0}.",__FUNCTION__ );
//    space_x->setEnabled(true);
//    space_y->setEnabled(true);
//    space_z->setEnabled(true);
    trivalVolume.reset();
    rawVolume.reset();
    tf_editor->resetTransferFunction();
    tf_editor->setEnabled(false);
    tf.clear();
    renderPolicy.clear();
//    render_policy_editor->volumeClosed();
}

//void SettingWidget::getSpace(float* space){
//    space[0] = space_x->value();
//    space[1] = space_y->value();
//    space[2] = space_z->value();
//}

//void SettingWidget::getRenderPolicy(float* rp){
//    for(int i = 0;i < renderPolicy.size();i++){
//        rp[i] = renderPolicy[i];
//    }
////    this->render_policy_editor->getRenderPolicy(rp);
//}