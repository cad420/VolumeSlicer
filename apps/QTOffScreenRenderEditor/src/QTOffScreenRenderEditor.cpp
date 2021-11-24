//
// Created by csh on 10/20/2021.
//

#include "QTOffScreenRenderEditor.h"
QTOffScreenRenderEditor::QTOffScreenRenderEditor(QWidget* parent):QMainWindow(parent){
    setWindowTitle("OffScreenRender");
    resize(1600,900);

    setWindowFlags(Qt::WindowCloseButtonHint | Qt::MSWindowsFixedSizeDialogHint);

    CreateAction();
    CreateMenu();
    CreateWidgets();
}

void QTOffScreenRenderEditor::open(std::string filename){
    setting_widget->volumeLoaded(filename);
//    std::ifstream in(filename);
//
//    if(!in.is_open()){
//        QMessageBox::critical(NULL,"Error","File open failed!",QMessageBox::Yes);
//        return;
//    }
//
//    try{
//        nlohmann::json j;
//        in>>j;
//
//        auto comp_volume_info=j["comp_volume"];
//        std::string comp_volume_path=comp_volume_info.at("comp_config_file_path");
//        volume_render_widget->loadVolume(comp_volume_path);
////        offscreen_render_setting_widget->volumeLoaded(comp_volume_path);
//
//        auto raw_volume_info=j["raw_volume"];
//        std::string raw_volume_path=raw_volume_info.at("raw_volume_path");
//        auto raw_volume_dim=raw_volume_info.at("raw_volume_dim");
//        auto raw_volume_space=raw_volume_info.at("raw_volume_space");
//        uint32_t dim_x=raw_volume_dim.at(0);
//        uint32_t dim_y=raw_volume_dim.at(1);
//        uint32_t dim_z=raw_volume_dim.at(2);
//        float space_x=raw_volume_space.at(0);
//        float space_y=raw_volume_space.at(1);
//        float space_z=raw_volume_space.at(2);
//        setting_widget->volumeLoaded(filename,comp_volume_path,raw_volume_path, dim_x, dim_y, dim_z, space_x, space_y, space_z);
//    }
//    catch (const std::exception& err) {
//        QMessageBox::critical(NULL,"Error","Config file format error!",QMessageBox::Yes);
//    }

    volume_render_widget->draw();
}

void QTOffScreenRenderEditor::CreateAction()
{
    open_file_action = new QAction(QIcon("../icons/open.png"),
                                   QStringLiteral("open"));
    connect(open_file_action,&QAction::triggered,[this](){
      this->open(
          QFileDialog::getOpenFileName(this,
                                       QStringLiteral("OpenFile"),
                                       QStringLiteral("."),
                                       QStringLiteral("config files(*.json)")
          ).toStdString()
      );
    });

}

void QTOffScreenRenderEditor::CreateMenu(){
    file_menu = menuBar()->addMenu("File");
    file_menu->addAction(open_file_action);
    file_menu->addAction(tr("Close"),this,[this](){
        PrintCUDAMemInfo("start close");
        setting_widget->volumeClosed();
        PrintCUDAMemInfo("after setting_widget volumeClose");
        volume_render_widget->volumeClosed();
        PrintCUDAMemInfo("after volume_render_widget volumeClose");
//        offscreen_render_setting_widget->volumeClosed();
//        PrintCUDAMemInfo("after offscreen_render_setting_widget volumeClose");
    });
    view_menu = menuBar()->addMenu("View");
}

void QTOffScreenRenderEditor::CreateWidgets()
{
    setDockOptions(QMainWindow::AnimatedDocks);
    setDockOptions(QMainWindow::AllowNestedDocks);
    setDockOptions(QMainWindow::AllowTabbedDocks);
    setDockNestingEnabled(true);

    volume_render_widget = new VolumeRenderWidget(this);
    volume_render_dock_widget = new QDockWidget(QStringLiteral("volume"));
    volume_render_dock_widget->setWidget(volume_render_widget);
    volume_render_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    volume_render_dock_widget->setFixedSize(700,1000);
    addDockWidget(Qt::LeftDockWidgetArea, volume_render_dock_widget);
    view_menu->addAction(volume_render_dock_widget->toggleViewAction());

    setting_widget = new SettingWidget(volume_render_widget, this);
    setting_dock_widget = new QDockWidget(QStringLiteral("real time render setting"));
    setting_dock_widget->setWidget(setting_widget);
    setting_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    setting_dock_widget->setFixedWidth(400);
    addDockWidget(Qt::LeftDockWidgetArea,setting_dock_widget);
    view_menu->addAction(setting_dock_widget->toggleViewAction());

    splitDockWidget(setting_dock_widget, volume_render_dock_widget, Qt::Horizontal);

//    offscreen_render_setting_widget = new OffScreenRenderSettingWidget(volume_render_widget, setting_widget, this);
//    offscreen_render_setting_dock_widget = new QDockWidget(QStringLiteral("offscreen render setting"));
//    offscreen_render_setting_dock_widget->setWidget(offscreen_render_setting_widget);
//    offscreen_render_setting_dock_widget->setAllowedAreas(Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
//    offscreen_render_setting_dock_widget->setFixedWidth(400);
//    addDockWidget(Qt::RightDockWidgetArea,offscreen_render_setting_dock_widget);
//    view_menu->addAction(offscreen_render_setting_dock_widget->toggleViewAction());

    volume_render_widget->setWidget(setting_widget);
}