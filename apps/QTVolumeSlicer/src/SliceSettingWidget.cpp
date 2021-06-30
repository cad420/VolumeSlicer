//
// Created by wyz on 2021/6/28.
//
#include "SliceSettingWidget.hpp"
#include <QPushButton>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QLabel>
#include <QtWidgets>
SliceSettingWidget::SliceSettingWidget(SliceRenderWidget *widget, QWidget *parent) {

    auto widget_layout=new QVBoxLayout;
    auto groupbox_layout=new QVBoxLayout;

    auto slice_args=new QGroupBox("Slice Setting");
    slice_args->setLayout(groupbox_layout);


    auto origin_label=new QLabel("Origin");
    auto origin_x_spin_box=new QDoubleSpinBox;
    auto origin_y_spin_box=new QDoubleSpinBox;
    auto origin_z_spin_box=new QDoubleSpinBox;


    groupbox_layout->addWidget(origin_label);
    groupbox_layout->setStretchFactor(origin_label,1);


    auto offset_label=new QLabel("Offset");
    groupbox_layout->addWidget(offset_label);
    auto offset_layout=new QHBoxLayout;
    groupbox_layout->addLayout(offset_layout);
    auto offset_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    offset_layout->addWidget(offset_horizontal_slider);
    auto offset_spin_box=new QDoubleSpinBox();
    offset_layout->addWidget(offset_spin_box);
    groupbox_layout->setStretchFactor(offset_label,1);
    groupbox_layout->setStretchFactor(offset_layout,1);


    auto rotation_label=new QLabel("Rotation");
    groupbox_layout->addWidget(rotation_label);
    groupbox_layout->setStretchFactor(rotation_label,1);

    auto reset_label=new QLabel("Reset");
    auto reset_combobox=new QComboBox;
    reset_combobox->addItem("x-axis");
    reset_combobox->addItem("y-axis");
    reset_combobox->addItem("z-axis");
    auto reset_layout=new QHBoxLayout;
    reset_layout->addWidget(reset_label);
    reset_layout->addWidget(reset_combobox);
    reset_layout->setStretchFactor(reset_label,1);
    reset_layout->setStretchFactor(reset_combobox,5);
    groupbox_layout->addLayout(reset_layout);
    groupbox_layout->setStretchFactor(reset_layout,1);

    auto lr_label=new QLabel("LR");
    auto lr_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    auto lr_spin_box=new QDoubleSpinBox();
    auto lr_layout=new QHBoxLayout;
    lr_layout->addWidget(lr_label);
    lr_layout->addWidget(lr_horizontal_slider);
    lr_layout->addWidget(lr_spin_box);
    groupbox_layout->addLayout(lr_layout);
    groupbox_layout->setStretchFactor(lr_layout,1);

    auto fb_label=new QLabel("FB");
    auto fb_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    auto fb_spin_box=new QDoubleSpinBox();
    auto fb_layout=new QHBoxLayout;
    fb_layout->addWidget(fb_label);
    fb_layout->addWidget(fb_horizontal_slider);
    fb_layout->addWidget(fb_spin_box);
    groupbox_layout->addLayout(fb_layout);
    groupbox_layout->setStretchFactor(fb_layout,1);

    auto ud_label=new QLabel("UD");
    auto ud_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    auto ud_spin_box=new QDoubleSpinBox();
    auto ud_layout=new QHBoxLayout;
    ud_layout->addWidget(ud_label);
    ud_layout->addWidget(ud_horizontal_slider);
    ud_layout->addWidget(ud_spin_box);
    groupbox_layout->addLayout(ud_layout);
    groupbox_layout->setStretchFactor(ud_layout,1);

    widget_layout->addWidget(slice_args);
    this->setLayout(widget_layout);
}
