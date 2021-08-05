//
// Created by wyz on 2021/6/28.
//
#include "VolumeRenderSettingWidget.hpp"
#include "VolumeRenderWidget.hpp"
#include "TransferFunctionWidget.hpp"
#include "tf1deditor.h"
#include <QVBoxLayout>
#include <QCheckBox>
#include <iostream>
#include <QScrollArea>
#include "TrivalVolume.hpp"

VolumeRenderSettingWidget::VolumeRenderSettingWidget(VolumeRenderWidget *widget, QWidget *parent)
:m_volume_render_widget(widget)
{
    auto widget_layout=new QVBoxLayout;

    auto visible_layout=new QHBoxLayout;
    auto volume_visible_check_box=new QCheckBox("volume");
    volume_visible_check_box->setChecked(true);
    auto slice_visible_check_box=new QCheckBox("slice");
    slice_visible_check_box->setChecked(true);
    connect(volume_visible_check_box,&QCheckBox::stateChanged,
            [this,volume_visible_check_box,slice_visible_check_box](int state){
        bool volume_visible=volume_visible_check_box->isChecked();
        bool slice_visible=slice_visible_check_box->isChecked();
        m_volume_render_widget->setVisible(volume_visible,slice_visible);
    });
    connect(slice_visible_check_box,&QCheckBox::stateChanged,
            [this,volume_visible_check_box,slice_visible_check_box](int state){
        bool volume_visible=volume_visible_check_box->isChecked();
        bool slice_visible=slice_visible_check_box->isChecked();
        std::cout<<"slice change: "<<slice_visible<<std::endl;
        m_volume_render_widget->setVisible(volume_visible,slice_visible);
    });

    visible_layout->addWidget(volume_visible_check_box);
    visible_layout->addWidget(slice_visible_check_box);

    widget_layout->addLayout(visible_layout);

//    tf_widget=new TransferFunctionWidget();
//    widget_layout->addWidget(tf_widget);

    tf_editor_widget=new TF1DEditor();

    auto raw_volume=m_volume_render_widget->getRawVolume();
    if(raw_volume){
        trival_volume=std::make_unique<TrivalVolume>(raw_volume->GetData(),raw_volume->GetVolumeDimX(),
                                                     raw_volume->GetVolumeDimY(),raw_volume->GetVolumeDimZ());
        tf_editor_widget->setVolumeInformation(trival_volume.get());
        tf_editor_widget->setFixedHeight(400);
        tf.resize(256*4,0.f);
    }

    connect(tf_editor_widget,&TF1DEditor::TF1DChanged,[this](){
        tf_editor_widget->getTransferFunction(tf.data(),256,1.0);
        m_volume_render_widget->resetTransferFunc1D(tf.data(),256);
        m_volume_render_widget->redraw();
    });
    widget_layout->addWidget(tf_editor_widget);


    this->setLayout(widget_layout);
}
void VolumeRenderSettingWidget::volumeLoaded() {
    auto raw_volume=m_volume_render_widget->getRawVolume();
    if(raw_volume){
        trival_volume=std::make_unique<TrivalVolume>(raw_volume->GetData(),raw_volume->GetVolumeDimX(),
                                                     raw_volume->GetVolumeDimY(),raw_volume->GetVolumeDimZ());
        tf_editor_widget->setVolumeInformation(trival_volume.get());
        tf_editor_widget->setFixedHeight(400);
        tf.resize(256*4,0.f);
    }
}
void VolumeRenderSettingWidget::volumeClose() {
    spdlog::info("{0}.",__FUNCTION__ );
    trival_volume.reset();
}