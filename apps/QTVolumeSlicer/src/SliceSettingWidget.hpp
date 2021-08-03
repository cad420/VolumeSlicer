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

class SliceSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceSettingWidget(SliceRenderWidget* widget,QWidget* parent= nullptr);

Q_SIGNALS:
    void sliceModified();

public Q_SLOTS:
    void updateSliceSettings();
private:
    void updateOrigin();
    void updateOffset();
    void updateNormal();

private:
    QDoubleSpinBox* origin_x_spin_box ;
    QDoubleSpinBox* origin_y_spin_box ;
    QDoubleSpinBox* origin_z_spin_box ;
    QSlider* offset_horizontal_slider;
    QDoubleSpinBox* offset_spin_box;
    int last_slider_value;
    std::array<float,3> last_slice_normal;
    float last_offset_length;
    float last_origin_offset;

    QScrollArea* m_slice_setting_scroll_area;
    SliceRenderWidget* m_slice_render_widget;
    TF1DEditor* tf_editor_widget;
    std::vector<float> tf;
    std::shared_ptr<Slicer> slicer;
    float space_x,space_y,space_z;
    float space_ratio_x,space_ratio_y,space_ratio_z;
    float volume_board_x,volume_board_y,volume_board_z;
};


#endif //VOLUMESLICER_SLICESETTINGWIDGET_HPP
