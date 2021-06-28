//
// Created by wyz on 2021/6/28.
//

#ifndef VOLUMESLICER_SLICESETTINGWIDGET_HPP
#define VOLUMESLICER_SLICESETTINGWIDGET_HPP

#include<QtWidgets/QWidget>

class SliceRenderWidget;

class SliceSettingWidget: public QWidget{
public:
    explicit SliceSettingWidget(SliceRenderWidget* widget,QWidget* parent= nullptr);
};


#endif //VOLUMESLICER_SLICESETTINGWIDGET_HPP
