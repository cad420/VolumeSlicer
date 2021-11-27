//
// Created by csh on 10/20/2021.
//
#include <VolumeSlicer/cuda_context.hpp>
#include "QTOffScreenRenderEditor.hpp"
QTOffScreenRenderEditor::QTOffScreenRenderEditor(QWidget* parent):QMainWindow(parent){
    setWindowTitle("OffScreen Render Editor");
    resize(1600,900);

    setWindowFlags(Qt::MSWindowsFixedSizeDialogHint);

    createAction();
    createMenu();
    createWidgets();
    SetCUDACtx(0);
}

void QTOffScreenRenderEditor::open(std::string filename){

    volume_render_widget->loadVolume(filename);
    off_setting_widget->setLoadedVolumeFile(filename);
}
void QTOffScreenRenderEditor::close()
{
    volume_render_widget->closeVolume();
}
void QTOffScreenRenderEditor::createAction()
{
    open_file_action = new QAction(QIcon("../icons/open.png"),
                                   QStringLiteral("Open"));
    connect(open_file_action,&QAction::triggered,this,[this](){
      this->open(
          QFileDialog::getOpenFileName(this,
                                       QStringLiteral("Load Volume"),
                                       QStringLiteral("."),
                                       QStringLiteral("config files(*.json)")
          ).toStdString()
      );
    });

    close_file_action = new QAction(QIcon(),QStringLiteral("Close"));

    connect(close_file_action,&QAction::triggered,this,[this](){
        this->close();
    });
}

void QTOffScreenRenderEditor::createMenu(){
    file_menu = menuBar()->addMenu("File");
    file_menu->addAction(open_file_action);
    file_menu->addAction(close_file_action);
    view_menu = menuBar()->addMenu("View");
}

void QTOffScreenRenderEditor::createWidgets()
{
    setDockOptions(QMainWindow::AnimatedDocks);
    setDockOptions(QMainWindow::AllowNestedDocks);
    setDockOptions(QMainWindow::AllowTabbedDocks);
    setDockNestingEnabled(true);

    volume_render_widget = new RealTimeVolumeRenderWidget(this);
    volume_render_dock_widget = new QDockWidget(QStringLiteral("real-time volume render"));
    volume_render_dock_widget->setWidget(volume_render_widget);
    volume_render_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    volume_render_dock_widget->setFixedSize(700,700);
    addDockWidget(Qt::LeftDockWidgetArea, volume_render_dock_widget);
    view_menu->addAction(volume_render_dock_widget->toggleViewAction());

    setting_widget = new RealTimeRenderSettingWidget(this);
    setting_dock_widget = new QDockWidget(QStringLiteral("real-time render setting"));
    setting_dock_widget->setWidget(setting_widget);
    setting_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
    setting_dock_widget->setFixedWidth(400);
    setting_dock_widget->setFixedHeight(500);
    addDockWidget(Qt::LeftDockWidgetArea,setting_dock_widget);
    view_menu->addAction(setting_dock_widget->toggleViewAction());

    off_setting_widget = new OffScreenRenderSettingWidget(this);
    off_setting_dock_widget = new QDockWidget(QStringLiteral("off-screen render setting"));
    off_setting_dock_widget->setWidget(off_setting_widget);
    off_setting_dock_widget->setAllowedAreas(Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
    off_setting_dock_widget->setFixedWidth(500);
    addDockWidget(Qt::RightDockWidgetArea,off_setting_dock_widget);
    view_menu->addAction(off_setting_dock_widget->toggleViewAction());

    splitDockWidget(setting_dock_widget, volume_render_dock_widget, Qt::Horizontal);


    connect(volume_render_widget,&RealTimeVolumeRenderWidget::volumeLoaded,this,[this](std::shared_ptr<CompVolume> comp_volume){
        this->setting_widget->volumeLoaded(comp_volume);
        this->setting_widget->resetTransferFunc();
        this->setting_widget->resetSteps();
        this->off_setting_widget->volumeLoaded(comp_volume);
        std::function<TransferFunc()> handle = [this](){
            return this->setting_widget->getTransferFunc();
        };
        this->off_setting_widget->setTransferFuncHandle(handle);
    });
    connect(volume_render_widget,&RealTimeVolumeRenderWidget::volumeClosed,this,[this](){
        this->setting_widget->volumeClosed();
        this->off_setting_widget->volumeClosed();
    });
    connect(setting_widget,&RealTimeRenderSettingWidget::StartingRecord,this,[this](){
       this->volume_render_widget->startRecording();
    });
    connect(setting_widget,&RealTimeRenderSettingWidget::StoppedRecord,this,[this](){
        this->volume_render_widget->stopRecording();
    });
    connect(volume_render_widget,&RealTimeVolumeRenderWidget::recordingStart,this,[this](){
        this->setting_widget->receiveRecordStarted();
    });
    connect(setting_widget,&RealTimeRenderSettingWidget::updateTransferFunc,this,[this](float* data,int dim){
       this->volume_render_widget->updateTransferFunc(data,dim);
    });
    connect(setting_widget,&RealTimeRenderSettingWidget::updateSteps,this,[this](int steps){
        this->volume_render_widget->updateSteps(steps);
    });
    connect(volume_render_widget,&RealTimeVolumeRenderWidget::recordingFinish,this,[this](std::vector<Camera> cameras){
        this->off_setting_widget->receiveRecordCameras(cameras);
    });

    OffScreenRenderSettingWidget::Handle tf_handle = [this](){
        return setting_widget->getTransferFunc();
    };
}

