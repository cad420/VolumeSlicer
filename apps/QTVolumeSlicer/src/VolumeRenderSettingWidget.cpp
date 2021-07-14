//
// Created by wyz on 2021/6/28.
//
#include "VolumeRenderSettingWidget.hpp"
#include "VolumeRenderWidget.hpp"
#include "TransferFunctionWidget.hpp"
#include <QVBoxLayout>
#include <QCheckBox>
#include <iostream>
#include <QScrollArea>

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

    tf_widget=new TransferFunctionWidget();
    widget_layout->addWidget(tf_widget);

    this->setLayout(widget_layout);
}
