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

    setting_widget = new SettingWidget(volume_render_widget,this);
    setting_dock_widget = new QDockWidget(QStringLiteral("real time render setting"));
    setting_dock_widget->setWidget(setting_widget);
    setting_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    setting_dock_widget->setFixedWidth(400);
    addDockWidget(Qt::LeftDockWidgetArea,setting_dock_widget);
    view_menu->addAction(setting_dock_widget->toggleViewAction());

    splitDockWidget(setting_dock_widget, volume_render_dock_widget, Qt::Horizontal);

    volume_render_widget->setWidget(setting_widget);
}