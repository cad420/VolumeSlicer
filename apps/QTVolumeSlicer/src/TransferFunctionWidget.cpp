//
// Created by wyz on 2021/6/30.
//
#include "TransferFunctionWidget.hpp"
#include <QColorDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSlider>
#include <QtWidgets>
#include "DoubleSlider.h"
#include "TFCanvas.hpp"
TransferFunctionWidget::TransferFunctionWidget(QWidget *parent) {
    auto widget_layout=new QVBoxLayout();

    auto buttons_layout=new QHBoxLayout();
    widget_layout->addLayout(buttons_layout);
    auto load_tf_button=new QPushButton(
            QIcon(
                            "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\QTVolumeSlicer\\icons\\TFOpen.png"
                    ),"Load transfer function file"
            );
    auto save_tf_button=new QPushButton(
            QIcon("C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\QTVolumeSlicer\\icons\\TFSave.png"),
            "Save transfer function to file"
            );
    buttons_layout->addWidget(load_tf_button);
    buttons_layout->addWidget(save_tf_button);

    auto tf_canvas=new TFCanvas();
    tf_canvas->setFixedSize(450,300);
    tf_canvas->setStyleSheet("background-color:white;");
    widget_layout->addWidget(tf_canvas);

    auto tf_range_double_slider=new DoubleSlider();
    widget_layout->addWidget(tf_range_double_slider);

    auto tf_range_spin_box_layout=new QHBoxLayout();
    auto tf_range_min_spin_box=new QDoubleSpinBox();
    auto tf_range_max_spin_box=new QDoubleSpinBox();
    tf_range_spin_box_layout->addWidget(tf_range_min_spin_box);
    tf_range_spin_box_layout->addWidget(tf_range_max_spin_box);
    widget_layout->addLayout(tf_range_spin_box_layout);

    this->setLayout(widget_layout);
}
