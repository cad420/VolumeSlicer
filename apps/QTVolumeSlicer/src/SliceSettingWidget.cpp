//
// Created by wyz on 2021/6/28.
//
#include "SliceSettingWidget.hpp"
#include "SliceRenderWidget.hpp"
#include "tf1deditor.h"
#include <iostream>

#include <QPushButton>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QLabel>
#include <QtWidgets>


SliceSettingWidget::SliceSettingWidget(SliceRenderWidget *widget, QWidget *parent)
: m_slice_render_widget(widget),update(false),
last_lr_slider_value(50),last_fb_slider_value(50),last_ud_slider_value(50),
last_lr_spin_value(0.0),last_fb_spin_value(0.0),last_ud_spin_value(0.0)
{
    auto widget_layout=new QVBoxLayout;

    auto groupbox_layout=new QVBoxLayout;

    auto slice_args=new QGroupBox("Slice Setting");
    slice_args->setLayout(groupbox_layout);


    auto origin_label=new QLabel("Origin");

    groupbox_layout->addWidget(origin_label);
    groupbox_layout->setStretchFactor(origin_label,1);


    origin_x_spin_box=new QDoubleSpinBox;
    origin_y_spin_box=new QDoubleSpinBox;
    origin_z_spin_box=new QDoubleSpinBox;
    origin_x_spin_box->setDecimals(3);
    origin_y_spin_box->setDecimals(3);
    origin_z_spin_box->setDecimals(3);
    origin_x_spin_box->setSingleStep(0.001);
    origin_y_spin_box->setSingleStep(0.001);
    origin_z_spin_box->setSingleStep(0.001);
    {
        auto comp_volume=m_slice_render_widget->getCompVolume();
        if(comp_volume){
            origin_x_spin_box->setMaximum(volume_board_x=(space_x=comp_volume->GetVolumeSpaceX())*comp_volume->GetVolumeDimX());
            origin_y_spin_box->setMaximum(volume_board_y=(space_y=comp_volume->GetVolumeSpaceY())*comp_volume->GetVolumeDimY());
            origin_z_spin_box->setMaximum(volume_board_z=(space_z=comp_volume->GetVolumeSpaceZ())*comp_volume->GetVolumeDimZ());
            float base_ratio=std::min({space_x,space_y,space_z});
            space_ratio_x=space_x/base_ratio;
            space_ratio_y=space_y/base_ratio;
            space_ratio_z=space_z/base_ratio;
        }
    }
    connect(origin_x_spin_box,&QDoubleSpinBox::valueChanged,[this](){
        if(update) return;
       auto slice=slicer->GetSlice();
       slice.origin[0]=origin_x_spin_box->value()/space_x;
       slicer->SetSlice(slice);
       slicer->SetStatus(true);
       emit sliceModified();
    });
    connect(origin_y_spin_box,&QDoubleSpinBox::valueChanged,[this](){
        if(update) return;
        auto slice=slicer->GetSlice();
        slice.origin[1]=origin_y_spin_box->value()/space_y;
        slicer->SetSlice(slice);
        slicer->SetStatus(true);
        emit sliceModified();
    });
    connect(origin_z_spin_box,&QDoubleSpinBox::valueChanged,[this](){
        if(update) return;
        auto slice=slicer->GetSlice();
        slice.origin[2]=origin_z_spin_box->value()/space_z;
        slicer->SetSlice(slice);
        slicer->SetStatus(true);
        emit sliceModified();
    });
    auto origin_layout=new QHBoxLayout;
    origin_layout->addWidget(origin_x_spin_box);
    origin_layout->addWidget(origin_y_spin_box);
    origin_layout->addWidget(origin_z_spin_box);
    groupbox_layout->addLayout(origin_layout);

    auto offset_label=new QLabel("Offset");
    groupbox_layout->addWidget(offset_label);
    auto offset_layout=new QHBoxLayout;
    groupbox_layout->addLayout(offset_layout);
    offset_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    offset_horizontal_slider->setSingleStep(1);
    offset_layout->addWidget(offset_horizontal_slider);
    offset_spin_box=new QDoubleSpinBox();
    offset_spin_box->setSingleStep(0.01);

    offset_layout->addWidget(offset_spin_box);
    groupbox_layout->setStretchFactor(offset_label,1);
    groupbox_layout->setStretchFactor(offset_layout,1);


    auto rotation_label=new QLabel("Rotation");
    groupbox_layout->addWidget(rotation_label);
    groupbox_layout->setStretchFactor(rotation_label,1);

    auto normal_label=new QLabel("Normal");

    normal_x_spin_box=new QDoubleSpinBox;
    normal_y_spin_box=new QDoubleSpinBox;
    normal_z_spin_box=new QDoubleSpinBox;

    normal_x_spin_box->setRange(-1.0,1.0);
    normal_y_spin_box->setRange(-1.0,1.0);
    normal_z_spin_box->setRange(-1.0,1.0);
    normal_x_spin_box->setDecimals(3);
    normal_y_spin_box->setDecimals(3);
    normal_z_spin_box->setDecimals(3);
    normal_x_spin_box->setSingleStep(0.1);
    normal_y_spin_box->setSingleStep(0.1);
    normal_z_spin_box->setSingleStep(0.1);

    auto normal_layout=new QHBoxLayout;
    normal_layout->addWidget(normal_label);
    normal_layout->addWidget(normal_x_spin_box);
    normal_layout->addWidget(normal_y_spin_box);
    normal_layout->addWidget(normal_z_spin_box);
    groupbox_layout->addLayout(normal_layout);

    auto reset_label=new QLabel("Reset Normal");
    auto reset_combobox=new QComboBox;
    reset_combobox->addItem("x-axis");
    reset_combobox->addItem("y-axis");
    reset_combobox->addItem("z-axis");
    auto reset_button=new QPushButton("reset");
    auto reset_layout=new QHBoxLayout;
    reset_layout->addWidget(reset_label);
    reset_layout->addWidget(reset_combobox);
    reset_layout->addWidget(reset_button);
    reset_layout->setStretchFactor(reset_label,1);
    reset_layout->setStretchFactor(reset_combobox,5);
    groupbox_layout->addLayout(reset_layout);
    groupbox_layout->setStretchFactor(reset_layout,1);

    auto lr_label=new QLabel("LR");
    lr_label->setFixedWidth(20);
    lr_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    lr_horizontal_slider->setRange(0,100);
    lr_horizontal_slider->setValue(50);
    lr_spin_box=new QDoubleSpinBox();
    lr_spin_box->setRange(-180.0,180);
    auto lr_layout=new QHBoxLayout;
    lr_layout->addWidget(lr_label);
    lr_layout->addWidget(lr_horizontal_slider);
    lr_layout->addWidget(lr_spin_box);
    groupbox_layout->addLayout(lr_layout);
    groupbox_layout->setStretchFactor(lr_layout,1);

    auto fb_label=new QLabel("FB");
    fb_label->setFixedWidth(20);
    fb_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    fb_horizontal_slider->setRange(0,100);
    fb_horizontal_slider->setValue(50);
    fb_spin_box=new QDoubleSpinBox();
    fb_spin_box->setRange(-180.0,180);
    auto fb_layout=new QHBoxLayout;
    fb_layout->addWidget(fb_label);
    fb_layout->addWidget(fb_horizontal_slider);
    fb_layout->addWidget(fb_spin_box);
    groupbox_layout->addLayout(fb_layout);
    groupbox_layout->setStretchFactor(fb_layout,1);

    auto ud_label=new QLabel("UD");
    ud_label->setFixedWidth(20);
    ud_horizontal_slider=new QSlider(Qt::Orientation::Horizontal);
    ud_horizontal_slider->setRange(0,100);
    ud_horizontal_slider->setValue(50);
    ud_spin_box=new QDoubleSpinBox();
    ud_spin_box->setRange(-180.0,180.0);
    auto ud_layout=new QHBoxLayout;
    ud_layout->addWidget(ud_label);
    ud_layout->addWidget(ud_horizontal_slider);
    ud_layout->addWidget(ud_spin_box);
    groupbox_layout->addLayout(ud_layout);
    groupbox_layout->setStretchFactor(ud_layout,1);

    slice_args->setFixedHeight(500);
    widget_layout->addWidget(slice_args);

    tf_editor_widget=new TF1DEditor;
    tf_editor_widget->setFixedHeight(300);
    widget_layout->addWidget(tf_editor_widget);
    tf.resize(256*4,0.f);
    connect(tf_editor_widget,&TF1DEditor::TF1DChanged,[this](){
        tf_editor_widget->getTransferFunction(tf.data(),256,1.0);
        m_slice_render_widget->resetColorTable(tf.data(),256);
        m_slice_render_widget->redraw();
    });
//    m_slice_setting_scroll_area=new QScrollArea(this);
//    m_slice_setting_scroll_area->setWidget(slice_args);
//    widget_layout->addWidget(m_slice_setting_scroll_area);
    this->setLayout(widget_layout);
    this->slicer=m_slice_render_widget->getSlicer();
    updateSliceSettings(false);
    connect(offset_horizontal_slider,&QSlider::valueChanged,[this](){
        if(update) return;
        spdlog::info("QSlider value changed:{0}.",offset_horizontal_slider->value());
        spdlog::info("last slider value:{0}.",last_slider_value);
        spdlog::info("last offset length:{0}.",last_offset_length);
        int cur_slider_value=offset_horizontal_slider->value();
        if(cur_slider_value==last_slider_value) return;
        float delta_offset=(cur_slider_value-last_slider_value)/100.f*last_offset_length;
        spdlog::info("delta_offset:{0}.",delta_offset);
        auto slice=slicer->GetSlice();

        slice.origin={slice.origin[0]+delta_offset*last_slice_normal[0]/space_x,
                      slice.origin[1]+delta_offset*last_slice_normal[1]/space_y,
                      slice.origin[2]+delta_offset*last_slice_normal[2]/space_z};

        slicer->SetSlice(slice);


//        updateOffset();
        last_slider_value=cur_slider_value;
        last_origin_offset=(cur_slider_value)/100.0*last_offset_length;
        this->offset_spin_box->setValue(last_origin_offset);

        emit sliceModified();
    });
    connect(offset_spin_box,&QDoubleSpinBox::valueChanged,[this](){
        if(update) return;
        auto slice=slicer->GetSlice();
        float cur_offset=offset_spin_box->value();
        float delta_offset=cur_offset-last_origin_offset;
        if(std::abs(delta_offset)<0.0001f) return;
        slice.origin={slice.origin[0]+delta_offset*last_slice_normal[0]/space_x,
                      slice.origin[1]+delta_offset*last_slice_normal[1]/space_y,
                      slice.origin[2]+delta_offset*last_slice_normal[2]/space_z};
        slicer->SetSlice(slice);

//        updateOffset();
        last_origin_offset=cur_offset;
        last_slider_value=cur_offset/last_offset_length*100;
        this->offset_horizontal_slider->setValue(last_slider_value);

        emit sliceModified();
    });
    //normal x
    connect(normal_x_spin_box,&QDoubleSpinBox::valueChanged,[this](double value){
        if(update) return;

        this->slicer->NormalIncreaseX(value-last_normal_x);

        this->update=true;
        initRotation();
        updateNormal();
        updateOffset();
        this->update=false;

        emit sliceModified();
    });
    //normal y
    connect(normal_y_spin_box,&QDoubleSpinBox::valueChanged,[this](double value){
       if(update) return;
        this->slicer->NormalIncreaseY(value-last_normal_y);

        this->update=true;
        initRotation();
        updateNormal();
        updateOffset();
        this->update=false;

        emit sliceModified();
    });
    //normal z
    connect(normal_z_spin_box,&QDoubleSpinBox::valueChanged,[this](double value){
        if(update) return;

        this->slicer->NormalIncreaseZ(value-last_normal_z);

        this->update=true;
        initRotation();
        updateNormal();
        updateOffset();
        this->update=false;

        emit sliceModified();
    });

    //reset button
    connect(reset_button,&QPushButton::clicked,[reset_combobox,this](){
        auto item=reset_combobox->currentText().toStdString();
        if(item=="x-axis"){
            slicer->NormalX();
        }
        else if(item=="y-axis"){
            slicer->NormalY();
        }
        else if(item=="z-axis"){
            slicer->NormalZ();
        }
        updateNormal();
        initRotation();
        emit sliceModified();
    });

    //lr
    //lr slider
    connect(lr_horizontal_slider,&QSlider::valueChanged,[this](int value){
        if(update) return;
        if(value==last_lr_slider_value ) return;

        double delta_degree=((value-last_lr_slider_value)*3.6)*3.141592627/180.0;
        spdlog::info("last:{1}, cur:{2}, lr delta degree: {0}.",delta_degree,last_lr_slider_value,value);
        this->slicer->RotateByX(delta_degree);

//        updateRotation();
        last_lr_slider_value=value;

        last_fb_slider_value=50;
        fb_horizontal_slider->setValue(last_fb_slider_value);
        last_ud_slider_value=50;
        ud_horizontal_slider->setValue(last_ud_slider_value);
        last_fb_spin_value=0.0;
        fb_spin_box->setValue(last_fb_spin_value);
        last_ud_spin_value=0.0;
        ud_spin_box->setValue(last_ud_spin_value);

        last_lr_spin_value=value*3.6-180.0;
        lr_spin_box->setValue(last_lr_spin_value);

        this->update=true;
        updateNormal();
        updateOffset();
        this->update=false;

        emit sliceModified();
    });
    //lr spin box
    connect(lr_spin_box,&QDoubleSpinBox::valueChanged,[this](double value){
        if(update) return;
        if(std::abs(last_lr_spin_value-value)<0.0001) return;

        last_lr_spin_value=value;

        lr_horizontal_slider->setValue((value+180.0)/3.6);

        this->update=true;
        updateNormal();
        updateOffset();
        this->update=false;
    });

    //fb
    //fb slider
    connect(fb_horizontal_slider,&QSlider::valueChanged,[this](int value){
        if(update) return;
        if(value==last_fb_slider_value) return;

        double delta_degree=((value-last_fb_slider_value)*3.6)*3.141592627/180.0;

        this->slicer->RotateByY(delta_degree);


        last_fb_slider_value=value;

        last_lr_slider_value=50;
        lr_horizontal_slider->setValue(last_lr_slider_value);
        last_ud_slider_value=50;
        ud_horizontal_slider->setValue(last_ud_slider_value);
        last_lr_spin_value=0.0;
        lr_spin_box->setValue(last_lr_spin_value);
        last_ud_spin_value=0.0;
        ud_spin_box->setValue(last_ud_spin_value);

        last_fb_spin_value=value*3.6-180.0;
        fb_spin_box->setValue(last_fb_spin_value);

        this->update=true;
        updateNormal();
        updateOffset();
        this->update=false;

        emit sliceModified();

    });
    //fb spin box
    connect(fb_spin_box,&QDoubleSpinBox::valueChanged,[this](double value){
        if(update) return;
        if(std::abs(last_fb_spin_value-value)<0.0001) return;

        last_fb_spin_value=value;

        fb_horizontal_slider->setValue((value+180.0)/3.6);

        this->update=true;
        updateNormal();
        updateOffset();
        this->update=false;
    });

    //ud
    //ud slider
    connect(ud_horizontal_slider,&QSlider::valueChanged,[this](int value){
        if(update) return;
        if(value==last_ud_slider_value) return;

        double delta_degree=((value-last_ud_slider_value)*3.6)*3.141592627/180.0;

        this->slicer->RotateByZ(delta_degree);

        last_ud_slider_value=value;

        last_lr_slider_value=50;
        lr_horizontal_slider->setValue(last_lr_slider_value);
        last_fb_slider_value=50;
        fb_horizontal_slider->setValue(last_fb_slider_value);
        last_fb_spin_value=0.0;
        fb_spin_box->setValue(last_fb_spin_value);
        last_ud_spin_value=0.0;
        ud_spin_box->setValue(last_ud_spin_value);

        last_ud_spin_value=value*3.6-180;
        ud_spin_box->setValue(last_ud_spin_value);

        this->update=true;
        updateNormal();
        updateOffset();
        this->update=false;

        emit sliceModified();
    });
    //ud spin box
    connect(ud_spin_box,&QDoubleSpinBox::valueChanged,[this](double value){
        if(update) return;
        if(std::abs(last_ud_spin_value-value)<0.0001) return;

        last_ud_spin_value=value;

        ud_horizontal_slider->setValue((value+180.0)/3.6);

        this->update=true;
        updateNormal();
        updateOffset();
        this->update=false;
    });
}
void SliceSettingWidget::updateSliceSettings(bool slice_update) {
    spdlog::info("update slice settings.");
    if(!slicer) return;
    this->update=slice_update;
    updateOrigin();
    updateOffset();
    updateNormal();
    updateRotation();
    this->update=false;
}

void SliceSettingWidget::updateOrigin() {
    auto slice=slicer->GetSlice();

    auto origin=slice.origin;
    origin_x_spin_box->setValue(origin[0]*space_x);
    origin_y_spin_box->setValue(origin[1]*space_y);
    origin_z_spin_box->setValue(origin[2]*space_z);
}

void SliceSettingWidget::updateOffset() {
    auto slice=slicer->GetSlice();
    auto origin=slice.origin;
    std::array<float,3> normal={slice.normal[0]*space_ratio_x,
                                slice.normal[1]*space_ratio_y,
                                slice.normal[2]*space_ratio_z};
    float length=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    normal={normal[0]/length,normal[1]/length,normal[2]/length};
    float offset_length=std::numeric_limits<float>::max();
    origin={origin[0]*space_x,origin[1]*space_y,origin[2]*space_z};
    float origin_offset=0.f;
    if(std::abs(normal[0])>0.00001f){
        float dist=std::abs(volume_board_x/normal[0]);
        if(dist<offset_length){
            offset_length=dist;
            origin_offset=origin[0]/std::abs(normal[0]);
        }
    }
    if(std::abs(normal[1])>0.00001f){
        float dist=std::abs(volume_board_y/normal[1]);
        if(dist<offset_length){
            offset_length=dist;
            origin_offset=origin[1]/std::abs(normal[1]);
        }
    }
    if(std::abs(normal[2])>0.00001f){
        float dist=std::abs(volume_board_z/normal[2]);
        if(dist<offset_length){
            offset_length=dist;
            origin_offset=origin[2]/std::abs(normal[2]);
        }
    }
    spdlog::info("volume board: {0} {1} {2}.",volume_board_x,volume_board_y,volume_board_z);
    spdlog::info("current offset length is: {0}.",offset_length);
    spdlog::info("current origin offset is: {0}.",origin_offset);
    last_slider_value=origin_offset/offset_length*100;
    last_slice_normal=normal;
    last_offset_length=offset_length;
    last_origin_offset=origin_offset;
    offset_horizontal_slider->setValue(last_slider_value);
    offset_spin_box->setValue(origin_offset);

}

void SliceSettingWidget::updateNormal() {
    auto slice=slicer->GetSlice();
    auto normal=slice.normal;
    this->normal_x_spin_box->setValue(normal[0]);
    this->normal_y_spin_box->setValue(normal[1]);
    this->normal_z_spin_box->setValue(normal[2]);
    last_normal_x=normal[0];
    last_normal_y=normal[1];
    last_normal_z=normal[2];
}

void SliceSettingWidget::updateRotation() {
    if(update) {
        initRotation();
        return;
    }
}
void SliceSettingWidget::initRotation() {
    this->lr_horizontal_slider->setValue(50);
    this->lr_spin_box->setValue(0.0);
    this->fb_horizontal_slider->setValue(50);
    this->fb_spin_box->setValue(0.0);
    this->ud_horizontal_slider->setValue(50);
    this->ud_spin_box->setValue(0.0);
}
void SliceSettingWidget::volumeLoaded() {
    auto comp_volume=m_slice_render_widget->getCompVolume();
    if(comp_volume){
        origin_x_spin_box->setMaximum(volume_board_x=(space_x=comp_volume->GetVolumeSpaceX())*comp_volume->GetVolumeDimX());
        origin_y_spin_box->setMaximum(volume_board_y=(space_y=comp_volume->GetVolumeSpaceY())*comp_volume->GetVolumeDimY());
        origin_z_spin_box->setMaximum(volume_board_z=(space_z=comp_volume->GetVolumeSpaceZ())*comp_volume->GetVolumeDimZ());
        float base_ratio=std::min({space_x,space_y,space_z});
        space_ratio_x=space_x/base_ratio;
        space_ratio_y=space_y/base_ratio;
        space_ratio_z=space_z/base_ratio;
    }
    this->slicer=m_slice_render_widget->getSlicer();
}