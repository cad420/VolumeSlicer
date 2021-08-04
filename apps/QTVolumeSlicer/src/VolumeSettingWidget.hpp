//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_VOLUMESETTINGWIDGET_HPP
#define VOLUMESLICER_VOLUMESETTINGWIDGET_HPP

#include<QtWidgets/QWidget>
class SliceRenderWidget;
class VolumeRenderWidget;

class VolumeSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit VolumeSettingWidget(SliceRenderWidget* sliceRenderWidget,
                                 VolumeRenderWidget* volumeRenderWidget,
                                 QWidget* parent= nullptr);
public Q_SLOTS:
    void volumeLoaded();
private:
    SliceRenderWidget* m_slice_render_widget;
    VolumeRenderWidget* m_volume_render_widget;
};

#endif //VOLUMESLICER_VOLUMESETTINGWIDGET_HPP
