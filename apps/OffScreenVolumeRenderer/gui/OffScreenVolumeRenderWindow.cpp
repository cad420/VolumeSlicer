//
// Created by wyz on 2021/11/15.
//
#include "OffScreenVolumeRenderWindow.hpp"
#include "RenderFrameWidget.hpp"
#include "CameraVisWidget.hpp"
#include "RenderProgressWidget.hpp"
#include <QDockWidget>
#include <QMenuBar>
#include <QFileDialog>
#include <VolumeSlicer/Utils/logger.hpp>
OffScreenVolumeRenderWindow::OffScreenVolumeRenderWindow()
:QMainWindow(nullptr)
{
    setWindowTitle("OffScreenVolumeRenderWindow");

    setFixedSize(600,700);
    createWidgets();
    createMenu();
}

OffScreenVolumeRenderWindow::OffScreenVolumeRenderWindow(const std::string& config_file)
:OffScreenVolumeRenderWindow()
{
    open(config_file);
    startRender();
}

void OffScreenVolumeRenderWindow::open(const std::string &config_file)
{
    try{
        if(config_file.empty()) return;
        this->render_config = OffScreenVolumeRenderer::LoadRenderConfigFromFile(config_file.c_str());
        render_progress_widget->SetRenderConfig(config_file,render_config);
        loadCameras(render_config.camera_sequence_config);
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("open config file error: {0}",err.what());
    }
}

void OffScreenVolumeRenderWindow::startRender()
{
    try{
        render_progress_widget->render();
    }
    catch(const std::exception& err){
        LOG_ERROR("startRender error: {0}",err.what());
    }
}
void OffScreenVolumeRenderWindow::createWidgets()
{
    setDockOptions(QMainWindow::AnimatedDocks);
    setDockOptions(QMainWindow::AllowNestedDocks);
    setDockOptions(QMainWindow::AllowTabbedDocks);
    setDockNestingEnabled(true);

    render_frame_widget = new RenderFrameWidget(this);
    render_frame_dock_widget = new QDockWidget(QStringLiteral("RenderFrame View"));
    render_frame_dock_widget->setWidget(render_frame_widget);
    render_frame_dock_widget->setMinimumSize(600,500);
    addDockWidget(Qt::BottomDockWidgetArea,render_frame_dock_widget);

    camera_vis_widget = new CameraVisWidget(this);
    camera_vis_dock_widget = new QDockWidget(QStringLiteral("CameraVis View"));
    camera_vis_dock_widget->setWidget(camera_vis_widget);
    camera_vis_dock_widget->setMinimumSize(600,500);
    addDockWidget(Qt::BottomDockWidgetArea,camera_vis_dock_widget);

    render_progress_widget = new RenderProgressWidget(this);
    render_progress_dock_widget = new QDockWidget(QStringLiteral("RenderProgress View"));
    render_progress_dock_widget->setWidget(render_progress_widget);
//    render_progress_dock_widget->setMinimumSize(400,200);
//    render_progress_dock_widget->setMaximumSize(600,200);
    addDockWidget(Qt::TopDockWidgetArea,render_progress_dock_widget);
    tabifyDockWidget(render_frame_dock_widget,camera_vis_dock_widget);
    connect(render_progress_widget,&RenderProgressWidget::RenderFrameFinish,
            this,[this](RenderProgressWidget::Pack pack){
        render_frame_widget->UpdateRenderFrame(render_config.width,render_config.height,pack.second);
        camera_vis_widget->UpdateCameraIndex(pack.first);
    });
}
void OffScreenVolumeRenderWindow::createMenu()
{
    auto m_file_menu = menuBar()->addMenu("File");
    auto m_open_action = new QAction(
                             QIcon(),
                             QStringLiteral("Open")
                             );
    connect(m_open_action,&QAction::triggered,this,[this](){
      this->open(
          QFileDialog::getOpenFileName(this,
                                       QStringLiteral("Open Config File"),
                                       QStringLiteral("."),
                                       QStringLiteral("config file(*.json)")).toStdString()
      );
    });

    m_file_menu->addAction(m_open_action);

    auto m_view_menu = menuBar()->addMenu("View");
    m_view_menu->addAction(render_progress_dock_widget->toggleViewAction());
    m_view_menu->addAction(render_frame_dock_widget->toggleViewAction());
    m_view_menu->addAction(camera_vis_dock_widget->toggleViewAction());

    connect(render_progress_widget,&RenderProgressWidget::RenderStart,this,[=](){
       m_open_action->setEnabled(false);
    });
    connect(render_progress_widget,&RenderProgressWidget::RenderStop,this,[=](){
        m_open_action->setEnabled(true);
    });

}
void OffScreenVolumeRenderWindow::loadCameras(const std::string &camera_file)
{
    auto cameras = OffScreenVolumeRenderer::LoadCameraSequenceFromFile(camera_file.c_str());
    std::vector<CameraVisWidget::CameraPoint> camera_pts;
    camera_pts.reserve(cameras.size());
    for(auto& camera:cameras){
        camera_pts.emplace_back(camera.pos);
    }
    camera_vis_widget->SetCameraPoints(std::move(camera_pts));
}
