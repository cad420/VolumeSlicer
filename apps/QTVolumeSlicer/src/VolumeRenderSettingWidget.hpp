//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_VOLUMERENDERSETTINGWIDGET_HPP
#define VOLUMESLICER_VOLUMERENDERSETTINGWIDGET_HPP

#include<QtWidgets/QWidget>
class VolumeRenderWidget;

class VolumeRenderSettingWidget: public QWidget{
    Q_OBJECT
public:
    explicit VolumeRenderSettingWidget(VolumeRenderWidget* widget,QWidget* parent= nullptr);
};

#endif //VOLUMESLICER_VOLUMERENDERSETTINGWIDGET_HPP
