//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICESETTINGWIDGET_HPP
#define VOLUMESLICER_SLICESETTINGWIDGET_HPP

#include<QtWidgets/QWidget>
#include <VolumeSlicer/slice.hpp>
#include <VolumeSlicer/volume.hpp>
using namespace vs;
class QGroupBox;
class QScrollArea;
class QDoubleSpinBox;
class QSlider;
class SliceRenderWidget;
class TF1DEditor;
class VolumeRenderWidget;
class TrivalVolume;

class SliceSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceSettingWidget(SliceRenderWidget* widget,QWidget* parent= nullptr);
    void SetVolumeRenderWidget(VolumeRenderWidget* widget);
Q_SIGNALS:
    void sliceModified();

public Q_SLOTS:
    void updateSliceSettings(bool slice_update);
    void volumeLoaded();
    void volumeClose();
private:
    void updateOrigin();
    void updateOffset();
    void updateNormal();
    void updateRotation();
    void initRotation();
private:
    bool update;
    QDoubleSpinBox* origin_x_spin_box ;
    QDoubleSpinBox* origin_y_spin_box ;
    QDoubleSpinBox* origin_z_spin_box ;
    QSlider* offset_horizontal_slider;
    QDoubleSpinBox* offset_spin_box;
    int last_slider_value;
    std::array<float,3> last_slice_normal;
    float last_offset_length;
    float last_origin_offset;

    QDoubleSpinBox* normal_x_spin_box;
    QDoubleSpinBox* normal_y_spin_box;
    QDoubleSpinBox* normal_z_spin_box;
    double last_normal_x;
    double last_normal_y;
    double last_normal_z;

    QSlider* lr_horizontal_slider;
    QDoubleSpinBox* lr_spin_box;
    int last_lr_slider_value;
    double last_lr_spin_value;

    QSlider* fb_horizontal_slider;
    QDoubleSpinBox* fb_spin_box;
    int last_fb_slider_value;
    double last_fb_spin_value;

    QSlider* ud_horizontal_slider;
    QDoubleSpinBox* ud_spin_box;
    int last_ud_slider_value;
    double last_ud_spin_value;

    QScrollArea* m_slice_setting_scroll_area;
    SliceRenderWidget* m_slice_render_widget;
    VolumeRenderWidget* m_volume_render_widget;
    std::unique_ptr<TrivalVolume> trival_volume;
    TF1DEditor* tf_editor_widget;
    std::vector<float> tf;
    std::shared_ptr<Slicer> slicer;
    float space_x,space_y,space_z;
    float space_ratio_x,space_ratio_y,space_ratio_z;
    float volume_board_x,volume_board_y,volume_board_z;
};


#endif //VOLUMESLICER_SLICESETTINGWIDGET_HPP
