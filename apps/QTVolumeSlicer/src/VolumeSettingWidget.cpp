//
// Created by wyz on 2021/6/28.
//
#include "VolumeSettingWidget.hpp"
#include "SliceRenderWidget.hpp"
#include "VolumeRenderWidget.hpp"
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
//        auto volume_name_layout=new QHBoxLayout();
//        auto volume_name_label=new QLabel("Comp Volume File");
//        auto volume_name_line_edit=new QLineEdit();
//        volume_name_line_edit->setReadOnly(true);
//
//        volume_name_layout->addWidget(volume_name_label);
//        volume_name_layout->addWidget(volume_name_line_edit);
//        comp_volume_layout->addLayout(volume_name_layout);


        auto volume_dim_layout=new QHBoxLayout();
        auto volume_dim_label=new QLabel("Comp Volume Dim");
        comp_volume_dim_x_line_edit=new QLineEdit("0");
        comp_volume_dim_y_line_edit=new QLineEdit("0");
        comp_volume_dim_z_line_edit=new QLineEdit("0");
        comp_volume_dim_x_line_edit->setReadOnly(true);
        comp_volume_dim_y_line_edit->setReadOnly(true);
        comp_volume_dim_z_line_edit->setReadOnly(true);

        volume_dim_layout->addWidget(volume_dim_label);
        volume_dim_layout->addWidget(comp_volume_dim_x_line_edit);
        volume_dim_layout->addWidget(comp_volume_dim_y_line_edit);
        volume_dim_layout->addWidget(comp_volume_dim_z_line_edit);
        comp_volume_layout->addLayout(volume_dim_layout);

        auto volume_space_layout=new QHBoxLayout();
        auto volume_space_label=new QLabel("Space");
        comp_volume_space_x_line_edit=new QLineEdit();
        comp_volume_space_y_line_edit=new QLineEdit();
        comp_volume_space_z_line_edit=new QLineEdit();
        volume_space_layout->addWidget(volume_space_label);
        volume_space_layout->addWidget(comp_volume_space_x_line_edit);
        volume_space_layout->addWidget(comp_volume_space_y_line_edit);
        volume_space_layout->addWidget(comp_volume_space_z_line_edit);
        comp_volume_layout->addLayout(volume_space_layout);
    }


    auto raw_volume_group_box=new QGroupBox();
    auto raw_volume_layout=new QVBoxLayout();
    raw_volume_group_box->setLayout(raw_volume_layout);
    {
//        auto volume_name_layout=new QHBoxLayout();
//        auto volume_name_label=new QLabel("Raw Volume File");
//        auto volume_name_line_edit=new QLineEdit();
//        volume_name_line_edit->setReadOnly(true);
//
//        volume_name_layout->addWidget(volume_name_label);
//        volume_name_layout->addWidget(volume_name_line_edit);
//        raw_volume_layout->addLayout(volume_name_layout);

        auto volume_dim_layout=new QHBoxLayout();
        auto volume_dim_label=new QLabel("Raw Volume Dim");
        raw_volume_dim_x_line_edit=new QLineEdit("0");
        raw_volume_dim_y_line_edit=new QLineEdit("0");
        raw_volume_dim_z_line_edit=new QLineEdit("0");
        raw_volume_dim_x_line_edit->setReadOnly(true);
        raw_volume_dim_y_line_edit->setReadOnly(true);
        raw_volume_dim_z_line_edit->setReadOnly(true);

        volume_dim_layout->addWidget(volume_dim_label);
        volume_dim_layout->addWidget(raw_volume_dim_x_line_edit);
        volume_dim_layout->addWidget(raw_volume_dim_y_line_edit);
        volume_dim_layout->addWidget(raw_volume_dim_z_line_edit);
        raw_volume_layout->addLayout(volume_dim_layout);

        auto volume_space_layout=new QHBoxLayout();
        auto volume_space_label=new QLabel("Space");
        raw_volume_space_x_line_edit=new QLineEdit();
        raw_volume_space_y_line_edit=new QLineEdit();
        raw_volume_space_z_line_edit=new QLineEdit();
        volume_space_layout->addWidget(volume_space_label);
        volume_space_layout->addWidget(raw_volume_space_x_line_edit);
        volume_space_layout->addWidget(raw_volume_space_y_line_edit);
        volume_space_layout->addWidget(raw_volume_space_z_line_edit);
        raw_volume_layout->addLayout(volume_space_layout);
    }
    comp_volume_group_box->setFixedHeight(200);
    raw_volume_group_box->setFixedHeight(200);
    widget_layout->addWidget(comp_volume_group_box);
    widget_layout->addWidget(raw_volume_group_box);
    this->setLayout(widget_layout);
}
//volume info will not change until load new volume
void VolumeSettingWidget::volumeLoaded() {
    {
        auto comp_volume=m_slice_render_widget->getCompVolume();
        comp_volume_dim_x_line_edit->setText(std::to_string(comp_volume->GetVolumeDimX()).c_str());
        comp_volume_dim_y_line_edit->setText(std::to_string(comp_volume->GetVolumeDimY()).c_str());
        comp_volume_dim_z_line_edit->setText(std::to_string(comp_volume->GetVolumeDimZ()).c_str());
        comp_volume_space_x_line_edit->setText(std::to_string(comp_volume->GetVolumeSpaceX()).c_str());
        comp_volume_space_y_line_edit->setText(std::to_string(comp_volume->GetVolumeSpaceY()).c_str());
        comp_volume_space_z_line_edit->setText(std::to_string(comp_volume->GetVolumeSpaceZ()).c_str());
    }
    {
        auto raw_volume=m_volume_render_widget->getRawVolume();
        raw_volume_dim_x_line_edit->setText(std::to_string(raw_volume->GetVolumeDimX()).c_str());
        raw_volume_dim_y_line_edit->setText(std::to_string(raw_volume->GetVolumeDimY()).c_str());
        raw_volume_dim_z_line_edit->setText(std::to_string(raw_volume->GetVolumeDimZ()).c_str());
        raw_volume_space_x_line_edit->setText(std::to_string(raw_volume->GetVolumeSpaceX()).c_str());
        raw_volume_space_y_line_edit->setText(std::to_string(raw_volume->GetVolumeSpaceY()).c_str());
        raw_volume_space_z_line_edit->setText(std::to_string(raw_volume->GetVolumeSpaceZ()).c_str());
    }
}