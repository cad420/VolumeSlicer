//
// Created by wyz on 2021/6/28.
//
#pragma once
#include<QtWidgets/QWidget>
class SliceRenderWidget;
class VolumeRenderWidget;

class QLineEdit;

class VolumeSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit VolumeSettingWidget(SliceRenderWidget* sliceRenderWidget,
                                 VolumeRenderWidget* volumeRenderWidget,
                                 QWidget* parent= nullptr);
public Q_SLOTS:
    void volumeLoaded();
    void volumeClose();
private:
    SliceRenderWidget* m_slice_render_widget;
    VolumeRenderWidget* m_volume_render_widget;

    QLineEdit* comp_volume_dim_x_line_edit;
    QLineEdit* comp_volume_dim_y_line_edit;
    QLineEdit* comp_volume_dim_z_line_edit;
    QLineEdit* comp_volume_space_x_line_edit;
    QLineEdit* comp_volume_space_y_line_edit;
    QLineEdit* comp_volume_space_z_line_edit;

    QLineEdit* raw_volume_dim_x_line_edit;
    QLineEdit* raw_volume_dim_y_line_edit;
    QLineEdit* raw_volume_dim_z_line_edit;
    QLineEdit* raw_volume_space_x_line_edit;
    QLineEdit* raw_volume_space_y_line_edit;
    QLineEdit* raw_volume_space_z_line_edit;

};

