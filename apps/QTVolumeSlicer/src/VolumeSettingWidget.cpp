//
// Created by wyz on 2021/6/28.
//
#include "VolumeSettingWidget.hpp"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
VolumeSettingWidget::VolumeSettingWidget(SliceRenderWidget *sliceRenderWidget,
                                         VolumeRenderWidget *volumeRenderWidget,
                                         QWidget *parent)
:m_slice_render_widget(sliceRenderWidget),m_volume_render_widget(volumeRenderWidget),
QWidget(parent)
{
    auto widget_layout=new QVBoxLayout();
    auto comp_volume_group_box=new QGroupBox("Comp Volume");
    auto comp_volume_layout=new QVBoxLayout();
    comp_volume_group_box->setLayout(comp_volume_layout);
    {
        auto volume_name_layout=new QHBoxLayout();
        auto volume_name_label=new QLabel("Comp Volume File");
        auto volume_name_line_edit=new QLineEdit();
        volume_name_line_edit->setReadOnly(true);
        auto volume_name_button=new QPushButton(QIcon(
                "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\QTVolumeSlicer\\icons\\open.png"
                ),"Load");
        connect(volume_name_button,&QPushButton::clicked,[this](){
            auto comp_volume_name=QFileDialog::getOpenFileName(this,
                                         QStringLiteral("OpenCompFile"),
                                         QStringLiteral("."),
                                         QStringLiteral("comp files(*.h264)")
            ).toStdString();
        });
        volume_name_layout->addWidget(volume_name_label);
        volume_name_layout->addWidget(volume_name_line_edit);
        volume_name_layout->addWidget(volume_name_button);
        comp_volume_layout->addLayout(volume_name_layout);


        auto volume_dim_layout=new QHBoxLayout();
        auto volume_dim_label=new QLabel("Comp Volume Dim");
        auto volume_dim_x_label=new QLabel("0");
        auto volume_dim_y_label=new QLabel("0");
        auto volume_dim_z_label=new QLabel("0");
        volume_dim_layout->addWidget(volume_dim_label);
        volume_dim_layout->addWidget(volume_dim_x_label);
        volume_dim_layout->addWidget(volume_dim_y_label);
        volume_dim_layout->addWidget(volume_dim_z_label);
        comp_volume_layout->addLayout(volume_dim_layout);

        auto volume_space_layout=new QHBoxLayout();
        auto volume_space_label=new QLabel("Space");
        auto volume_space_x_line_edit=new QLineEdit();
        auto volume_space_y_line_edit=new QLineEdit();
        auto volume_space_z_line_edit=new QLineEdit();
        volume_space_layout->addWidget(volume_space_label);
        volume_space_layout->addWidget(volume_space_x_line_edit);
        volume_space_layout->addWidget(volume_space_y_line_edit);
        volume_space_layout->addWidget(volume_space_z_line_edit);
        comp_volume_layout->addLayout(volume_space_layout);

    }


    auto raw_volume_group_box=new QGroupBox();
    auto raw_volume_layout=new QVBoxLayout();
    raw_volume_group_box->setLayout(raw_volume_layout);
    {
        auto volume_name_layout=new QHBoxLayout();
        auto volume_name_label=new QLabel("Comp Volume File");
        auto volume_name_line_edit=new QLineEdit();
        volume_name_line_edit->setReadOnly(true);
        auto volume_name_button=new QPushButton(QIcon(
                "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\QTVolumeSlicer\\icons\\open.png"
        ),"Load");
        connect(volume_name_button,&QPushButton::clicked,[this](){
            auto comp_volume_name=QFileDialog::getOpenFileName(this,
                                                               QStringLiteral("OpenRawFile"),
                                                               QStringLiteral("."),
                                                               QStringLiteral("raw files(*.raw)")
            ).toStdString();
        });
        volume_name_layout->addWidget(volume_name_label);
        volume_name_layout->addWidget(volume_name_line_edit);
        volume_name_layout->addWidget(volume_name_button);
        raw_volume_layout->addLayout(volume_name_layout);

        auto volume_dim_layout=new QHBoxLayout();
        auto volume_dim_label=new QLabel("Raw Volume Dim");
        auto volume_dim_x_label=new QLabel("0");
        auto volume_dim_y_label=new QLabel("0");
        auto volume_dim_z_label=new QLabel("0");
        volume_dim_layout->addWidget(volume_dim_label);
        volume_dim_layout->addWidget(volume_dim_x_label);
        volume_dim_layout->addWidget(volume_dim_y_label);
        volume_dim_layout->addWidget(volume_dim_z_label);
        raw_volume_layout->addLayout(volume_dim_layout);

        auto volume_space_layout=new QHBoxLayout();
        auto volume_space_label=new QLabel("Space");
        auto volume_space_x_line_edit=new QLineEdit();
        auto volume_space_y_line_edit=new QLineEdit();
        auto volume_space_z_line_edit=new QLineEdit();
        volume_space_layout->addWidget(volume_space_label);
        volume_space_layout->addWidget(volume_space_x_line_edit);
        volume_space_layout->addWidget(volume_space_y_line_edit);
        volume_space_layout->addWidget(volume_space_z_line_edit);
        raw_volume_layout->addLayout(volume_space_layout);
    }

    widget_layout->addWidget(comp_volume_group_box);
    widget_layout->addWidget(raw_volume_group_box);
    this->setLayout(widget_layout);
}
