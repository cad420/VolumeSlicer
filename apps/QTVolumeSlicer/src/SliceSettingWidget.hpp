//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICESETTINGWIDGET_HPP
#define VOLUMESLICER_SLICESETTINGWIDGET_HPP

#include<QtWidgets/QWidget>

class QGroupBox;
class QScrollArea;

class SliceRenderWidget;
class TF1DEditor;

class SliceSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit SliceSettingWidget(SliceRenderWidget* widget,QWidget* parent= nullptr);

private:
    QScrollArea* m_slice_setting_scroll_area;
    SliceRenderWidget* m_slice_render_widget;
    TF1DEditor* tf_editor_widget;
    std::vector<float> tf;
};


#endif //VOLUMESLICER_SLICESETTINGWIDGET_HPP
