#include "RealTimeRenderSettingWidget.hpp"
#include <QLabel>
#include <QDoubleSpinBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include "TrivalVolume.hpp"
#include "tf1deditor.h"
#include <VolumeSlicer/Utils/logger.hpp>
RealTimeRenderSettingWidget::RealTimeRenderSettingWidget(QWidget *parent)
:QWidget(parent)
{
    auto layout = new QVBoxLayout;


    volume_range_label = new QLabel("Volume Range: (0,0,0) - (0,0,0)");

    layout->addWidget(volume_range_label);

    auto camera_pos_layout = new QHBoxLayout();
    camera_pos_label = new QLabel("Camera Position ");
    camera_pos_x = new QDoubleSpinBox();
    camera_pos_y = new QDoubleSpinBox();
    camera_pos_z = new QDoubleSpinBox();
    camera_pos_x->setDecimals(3);
    camera_pos_x->setSingleStep(0.001);
    camera_pos_y->setDecimals(3);
    camera_pos_y->setSingleStep(0.001);
    camera_pos_z->setDecimals(3);
    camera_pos_z->setSingleStep(0.001);
    camera_pos_layout->addWidget(camera_pos_label);
    camera_pos_layout->addWidget(camera_pos_x);
    camera_pos_layout->addWidget(camera_pos_y);
    camera_pos_layout->addWidget(camera_pos_z);

    layout->addLayout(camera_pos_layout);

    auto steps_layout = new QHBoxLayout;
    steps_slider = new QSlider(Qt::Orientation::Horizontal);
    steps_sb = new QSpinBox;
    int min_val = 500,max_val =10000;
    steps_slider->setRange(min_val,max_val);
    steps_slider->setSingleStep(100);

    steps_sb->setRange(min_val,max_val);
    //enabled until volume loaded
    steps_slider->setEnabled(false);
    steps_sb->setEnabled(false);

    connect(steps_slider,&QSlider::valueChanged,this,[this](int value){
        steps_sb->setValue(value);
        resetSteps();
    });
    connect(steps_sb,&QSpinBox::valueChanged,this,[this](int value){
        steps_slider->setValue(value);
        resetSteps();
    });


    steps_layout->addWidget(steps_sb);
    steps_layout->addWidget(steps_slider);

    layout->addLayout(steps_layout);

    tf_editor_widget=new TF1DEditor;
    tf_editor_widget->setFixedHeight(300);
    layout->addWidget(tf_editor_widget);
    connect(tf_editor_widget,&TF1DEditor::TF1DChanged,this,[this](){
        this->resetTransferFunc();
    });

    auto record_layout = new QHBoxLayout;
    start_record_pb = new QPushButton("Start Record");
    stop_record_pb = new QPushButton("Stop Record");
    start_record_pb->setEnabled(!recording);
    stop_record_pb->setEnabled(recording);
    record_layout->addWidget(start_record_pb);
    record_layout->addWidget(stop_record_pb);

    layout->addLayout(record_layout);

    connect(start_record_pb,&QPushButton::clicked,this,[this](){
       startRecord();
    });
    connect(stop_record_pb,&QPushButton::clicked,this,[this](){
        stopRecord();
    });

    this->setLayout(layout);
}
void RealTimeRenderSettingWidget::volumeClosed()
{
    volume_range_label->setText("Volume Range: (0,0,0) - (0,0,0)");
    camera_pos_x->clear();
    camera_pos_y->clear();
    camera_pos_z->clear();
    steps_slider->setEnabled(false);
    steps_sb->setEnabled(false);
}
void RealTimeRenderSettingWidget::volumeLoaded(const std::shared_ptr<CompVolume>& comp_volume)
{
    auto volume_dim_x = comp_volume->GetVolumeDimX();
    auto volume_dim_y = comp_volume->GetVolumeDimY();
    auto volume_dim_z = comp_volume->GetVolumeDimZ();
    auto volume_space_x = comp_volume->GetVolumeSpaceX();
    auto volume_space_y = comp_volume->GetVolumeSpaceY();
    auto volume_space_z = comp_volume->GetVolumeSpaceZ();
    auto volume_range_x = volume_dim_x * volume_space_x;
    auto volume_range_y = volume_dim_y * volume_space_y;
    auto volume_range_z = volume_dim_z * volume_space_z;
    auto range = QString().asprintf("Volume Range : (0,0,0) - (%.2f,%.2f,%.2f)",
                                    volume_range_x,volume_range_y,volume_range_z);
    camera_pos_x->setRange(-volume_range_x,2*volume_range_x);
    camera_pos_y->setRange(-volume_range_y,2*volume_range_y);
    camera_pos_z->setRange(-volume_range_z,2*volume_range_z);
    volume_range_label->setText(range);

    //init steps for real-time renderer
    steps_slider->setEnabled(true);
    steps_sb->setEnabled(true);
    this->steps_sb->setValue((steps_sb->minimum()+steps_sb->maximum())/2);
}
void RealTimeRenderSettingWidget::startRecord()
{
    if(recording){
        LOG_ERROR("startRecord error: recording is on");
        return;
    }

    emit StartingRecord();

}
void RealTimeRenderSettingWidget::receiveRecordStarted()
{
    if(recording){
        LOG_ERROR("receiveRecordStarted error: recording is on already");
        return;
    }
    recording = true;
    updateRecordPB(true);
}
void RealTimeRenderSettingWidget::stopRecord()
{
    if(!recording){
        LOG_ERROR("stopRecord error: recording is off");
        return;
    }

    emit StoppedRecord();
    recording = false;
    updateRecordPB(false);
}
void RealTimeRenderSettingWidget::updateRecordPB(bool recording)
{
    start_record_pb->setEnabled(!recording);
    stop_record_pb->setEnabled(recording);
}
void RealTimeRenderSettingWidget::resetTransferFunc()
{
    static std::vector<float> data(256*4);
    tf_editor_widget->getTransferFunction(data.data(),256,1.0);
    emit updateTransferFunc(data.data(),256);
}
void RealTimeRenderSettingWidget::resetSteps()
{
    emit updateSteps(this->steps_sb->value());
}
auto RealTimeRenderSettingWidget::getTransferFunc() -> TransferFunc
{
    static std::vector<float> data(256*4);
    tf_editor_widget->getTransferFunction(data.data(),256,1.0);
    TransferFunc tf;
    for(uint8_t i=0;i<255;i++){
        tf.points.emplace_back(i,std::array<double,4>{(double)data[i*4+0],
                                                      (double)data[i*4+1],
                                                      (double)data[i*4+2],
                                                      (double)data[i*4+3]});
    }

    return tf;
}