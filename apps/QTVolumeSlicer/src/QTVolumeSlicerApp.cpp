//
// Created by wyz on 2021/6/11.
//
//app
#include "global.hpp"
#include "QTVolumeSlicerApp.hpp"
#include "VolumeRenderWidget.hpp"
#include "SliceRenderWidget.hpp"
#include "SliceZoomWidget.hpp"
#include "SliceSettingWidget.hpp"
#include "VolumeSettingWidget.hpp"
#include "VolumeRenderSettingWidget.hpp"
//vs
#include <VolumeSlicer/utils.hpp>
//std
#include <iostream>
#include <fstream>
//qt
#include <QMenuBar>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QApplication>
#include <QDockWidget>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QToolBar>
#include <QComboBox>
#include <QFileDialog>
#include <QMessageBox>
//json
#include "json.hpp"
using nlohmann::json;

VolumeSlicerMainWindow::VolumeSlicerMainWindow(QWidget *parent)
:QMainWindow(parent)
{
    setWindowTitle("VolumeSlicer");
    resize(1920,1080);
    createActions();
    createMenu();
    createWidgets();
    createToolBar();
}
void VolumeSlicerMainWindow::open(const std::string &file_name) {
    PrintCUDAMemInfo("start open");
    json j;
    std::ifstream in(file_name);
    if(!in.is_open()){
        QMessageBox::critical(NULL,"Error","File open failed!",QMessageBox::Yes);
        return;
    }
    try{
        in>>j;
        auto comp_volume_info=j["comp_volume"];
        std::string comp_volume_path=comp_volume_info.at("comp_config_file_path");
        auto comp_volume_space=comp_volume_info.at("comp_volume_space");
        float space_x=comp_volume_space.at(0);
        float space_y=comp_volume_space.at(1);
        float space_z=comp_volume_space.at(2);

        PrintCUDAMemInfo("in open before load comp volume");
        this->m_slice_render_widget->loadVolume(
                comp_volume_path.c_str(),
                {space_x,space_y,space_z}
        );
        PrintCUDAMemInfo("in open after load comp volume");

        auto raw_volume_info=j["raw_volume"];
        std::string raw_volume_path=raw_volume_info.at("raw_volume_path");
        auto raw_volume_dim=raw_volume_info.at("raw_volume_dim");
        auto raw_volume_space=raw_volume_info.at("raw_volume_space");
        uint32_t dim_x=raw_volume_dim.at(0);
        uint32_t dim_y=raw_volume_dim.at(1);
        uint32_t dim_z=raw_volume_dim.at(2);
        space_x=raw_volume_space.at(0);
        space_y=raw_volume_space.at(1);
        space_z=raw_volume_space.at(2);

        PrintCUDAMemInfo("in open before load raw volume");
        this->m_volume_render_widget->loadVolume(
                raw_volume_path.c_str(),
                {dim_x,dim_y,dim_z},{space_x,space_y,space_z}
        );
        PrintCUDAMemInfo("in open after load raw volume");
    }
    catch (const std::exception& err) {
        QMessageBox::critical(NULL,"Error","Config file format error!",QMessageBox::Yes);
    }

    PrintCUDAMemInfo("after open comp and raw volume");
    m_volume_render_widget->setSlicer(m_slice_render_widget->getSlicer());
    PrintCUDAMemInfo("after m_volume_render_widget setSlicer");
    m_slice_zoom_widget->setRawVolume(m_volume_render_widget->getRawVolume());
    PrintCUDAMemInfo("after m_slice_zoom_widget setRawVolume");
    m_slice_zoom_widget->setSlicer(m_slice_render_widget->getSlicer());
    PrintCUDAMemInfo("after m_slice_zoom_widget setSlicer");
    m_volume_setting_widget->volumeLoaded();
    PrintCUDAMemInfo("after m_volume_setting_widget volumeLoaded");
    m_volume_render_setting_widget->volumeLoaded();
    PrintCUDAMemInfo("after m_volume_render_setting_widget volumeLoaded");
    m_slice_setting_widget->volumeLoaded();
    PrintCUDAMemInfo("after m_slice_setting_widget volumeLoaded");
}
void VolumeSlicerMainWindow::createMenu() {
    m_file_menu=menuBar()->addMenu("File");

    m_file_menu->addAction(m_open_action);
    m_file_menu->addSeparator();

    m_file_menu->addAction(tr("Close"),this,[this](){
        PrintCUDAMemInfo("start close");
        m_slice_render_widget->volumeClose();
        PrintCUDAMemInfo("after m_slice_render_widget volumeClose");
        m_volume_render_widget->volumeClose();
        PrintCUDAMemInfo("after m_slice_render_widget volumeClose");
        m_slice_zoom_widget->volumeClose();
        PrintCUDAMemInfo("after m_slice_render_widget volumeClose");
        m_slice_setting_widget->volumeClose();
        PrintCUDAMemInfo("after m_slice_render_widget volumeClose");
        m_volume_render_setting_widget->volumeClose();
        PrintCUDAMemInfo("after m_slice_render_widget volumeClose");
        m_volume_setting_widget->volumeClose();
        PrintCUDAMemInfo("after m_slice_render_widget volumeClose");
    });
    m_view_menu=menuBar()->addMenu("View");

}

void VolumeSlicerMainWindow::createActions() {
    m_open_action=new QAction(QIcon(
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\QTVolumeSlicer\\icons\\open.png"
            ),QStringLiteral("Open"));
    m_open_action->setToolTip("Open raw/comp volume data");
    connect(m_open_action,&QAction::triggered,[this](){
        this->open(
                QFileDialog::getOpenFileName(this,
                                     QStringLiteral("OpenFile"),
                                     QStringLiteral("."),
                                     QStringLiteral("config files(*.json)")
                                     ).toStdString()
                                     );
    });

    m_volume_setting_action=new QAction(
            QIcon(),
            QStringLiteral("Volume")
            );

    m_slice_setting_action=new QAction(
            QIcon(),
            QStringLiteral("Slice")
            );
    m_volume_render_setting_action=new QAction(
            QIcon(),
            QStringLiteral("Volume Render")
            );

}
void VolumeSlicerMainWindow::createToolBar() {
    m_tool_bar=addToolBar(QStringLiteral("Tools"));
    m_tool_bar->addAction(m_open_action);
    m_module_panel=new QComboBox();

    m_module_panel->addItem("Volume");
    m_module_panel->addItem("Slice");
    m_module_panel->addItem("Volume Render");
    connect(m_module_panel,&QComboBox::currentTextChanged,[this](const QString& text){
        std::cout<<text.toStdString()<<std::endl;
        if(text=="Volume"){
            setModulePanel(m_volume_setting_widget);
        }
        else if(text=="Slice"){
            setModulePanel(m_slice_setting_widget);
        }
        else if(text=="Volume Render"){
            setModulePanel(m_volume_render_setting_widget);
        }
    });
    emit m_module_panel->currentTextChanged("Volume");
    m_tool_bar->addWidget(m_module_panel);



}
void VolumeSlicerMainWindow::setModulePanel(QWidget* widget) {
    m_setting_dock_widget->setWidget(widget);
}
void VolumeSlicerMainWindow::createWidgets() {
    setDockOptions(QMainWindow::AnimatedDocks);
    setDockOptions(QMainWindow::AllowNestedDocks);
    setDockOptions(QMainWindow::AllowTabbedDocks);
    setDockNestingEnabled(true);

    m_slice_render_widget=new SliceRenderWidget(this);
    m_slice_render_dock_widget=new QDockWidget(QStringLiteral("Slice View"));
    m_slice_render_dock_widget->setWidget(m_slice_render_widget);
    m_slice_render_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    m_slice_render_dock_widget->setMinimumSize(900,900);

    addDockWidget(Qt::LeftDockWidgetArea,m_slice_render_dock_widget);
    m_view_menu->addAction(m_slice_render_dock_widget->toggleViewAction());

    m_volume_render_widget=new VolumeRenderWidget(this);
    m_volume_render_dock_widget=new QDockWidget(QStringLiteral("Volume View"));
    m_volume_render_dock_widget->setWidget(m_volume_render_widget);
    m_volume_render_dock_widget->setAllowedAreas( Qt::RightDockWidgetArea | Qt::TopDockWidgetArea);
    m_volume_render_dock_widget->setMinimumSize(400,400);
    m_volume_render_dock_widget->setMaximumSize(900,900);
//    QSizePolicy qsp(QSizePolicy::Preferred,QSizePolicy::Preferred);
//    qsp.setHeightForWidth(true);
//    m_volume_render_widget->setSizePolicy(qsp);
//    m_volume_render_dock_widget->setSizePolicy(qsp);

    addDockWidget(Qt::RightDockWidgetArea,m_volume_render_dock_widget);
    m_view_menu->addAction(m_volume_render_dock_widget->toggleViewAction());



    m_slice_zoom_widget=new SliceZoomWidget(this);
    m_slice_zoom_dock_widget=new QDockWidget(QStringLiteral("Slice Zoom"));
    m_slice_zoom_dock_widget->setWidget(m_slice_zoom_widget);
    m_slice_zoom_dock_widget->setAllowedAreas( Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea );
    m_slice_zoom_dock_widget->setMaximumSize(400,400);
    m_slice_zoom_dock_widget->setMinimumSize(400,400);
    addDockWidget(Qt::RightDockWidgetArea,m_slice_zoom_dock_widget);
    m_view_menu->addAction(m_slice_zoom_dock_widget->toggleViewAction());

    splitDockWidget(m_volume_render_dock_widget,m_slice_zoom_dock_widget,Qt::Vertical);

    m_slice_setting_widget=new SliceSettingWidget(m_slice_render_widget,this);
    m_volume_render_setting_widget=new VolumeRenderSettingWidget(m_volume_render_widget,this);
    m_volume_setting_widget=new VolumeSettingWidget(m_slice_render_widget,
                                                    m_volume_render_widget,
                                                    this);
    //if slice setting changed, notice slice render to emit and redraw
    connect(m_slice_setting_widget,&SliceSettingWidget::sliceModified,[this](){
//       emit m_slice_render_widget->sliceModified();//m_slice_setting_widget will receive this signal
//       std::async(std::launch::async,&SliceRenderWidget::redraw,m_slice_render_widget);
//       std::async(std::launch::async,&SliceZoomWidget::redraw,m_slice_zoom_widget);
//       std::async(std::launch::async,&VolumeRenderWidget::redraw,m_volume_render_widget);
        m_slice_render_widget->redraw();
        m_volume_render_widget->redraw();
        m_slice_zoom_widget->redraw();
    });
    connect(m_slice_render_widget,&SliceRenderWidget::sliceModified,[this](){
        spdlog::info("slice render widget emit sliceModify.");
        m_slice_setting_widget->updateSliceSettings(true);
    });


    m_setting_dock_widget=new QDockWidget(QStringLiteral("Control Panned"),this);

    m_setting_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea);
    m_setting_dock_widget->setMinimumSize(500,900);
    m_setting_dock_widget->setMaximumSize(500,1200);
    m_view_menu->addAction(m_setting_dock_widget->toggleViewAction());
    addDockWidget(Qt::LeftDockWidgetArea,m_setting_dock_widget);

    splitDockWidget(m_setting_dock_widget,m_slice_render_dock_widget,Qt::Horizontal);


    {
//        m_volume_render_widget->setSlicer(m_slice_render_widget->getSlicer());
//        m_slice_zoom_widget->setRawVolume(m_volume_render_widget->getRawVolume());
//        m_slice_zoom_widget->setSlicer(m_slice_render_widget->getSlicer());
    }

    connect(m_slice_render_widget,&SliceRenderWidget::sliceModified,m_volume_render_widget,
            &VolumeRenderWidget::redraw);

    connect(m_slice_render_widget,&SliceRenderWidget::sliceModified,m_slice_zoom_widget,
            &SliceZoomWidget::redraw);
}




