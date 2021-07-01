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

VolumeSlicerMainWindow::VolumeSlicerMainWindow(QWidget *parent)
:QMainWindow(parent)
{
    setWindowTitle("VolumeSlicer");
    resize(1920,1080);
    createActions();
    createMenu();
    createWidgets();
    createToolBar();
    initTest();
}
void VolumeSlicerMainWindow::open(const std::string &file_name) {
    if(file_name.empty())
        return ;

}
void VolumeSlicerMainWindow::createMenu() {
    m_file_menu=menuBar()->addMenu("File");

    m_file_menu->addAction(m_open_action);
    m_file_menu->addSeparator();

    m_file_menu->addAction(tr("Close"),this,[](){
        std::cout<<"Close"<<std::endl;
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
                                     QStringLiteral("raw files(*.raw);;comp files(*.h264)")
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
    m_slice_render_dock_widget->setMinimumSize(800,800);
    addDockWidget(Qt::LeftDockWidgetArea,m_slice_render_dock_widget);
    m_view_menu->addAction(m_slice_render_dock_widget->toggleViewAction());

    m_volume_render_widget=new VolumeRenderWidget(this);
    m_volume_render_dock_widget=new QDockWidget(QStringLiteral("Volume View"));
    m_volume_render_dock_widget->setWidget(m_volume_render_widget);
    m_volume_render_dock_widget->setAllowedAreas( Qt::RightDockWidgetArea | Qt::TopDockWidgetArea);

    m_volume_render_dock_widget->setMaximumSize(500,500);
    addDockWidget(Qt::RightDockWidgetArea,m_volume_render_dock_widget);
    m_view_menu->addAction(m_volume_render_dock_widget->toggleViewAction());



    m_slice_zoom_widget=new SliceZoomWidget(this);
    m_slice_zoom_dock_widget=new QDockWidget(QStringLiteral("Slice Zoom"));
    m_slice_zoom_dock_widget->setWidget(m_slice_zoom_widget);
    m_slice_zoom_dock_widget->setAllowedAreas( Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea );
    m_slice_zoom_dock_widget->setMaximumSize(500,500);
    addDockWidget(Qt::RightDockWidgetArea,m_slice_zoom_dock_widget);
    m_view_menu->addAction(m_slice_zoom_dock_widget->toggleViewAction());

    splitDockWidget(m_volume_render_dock_widget,m_slice_zoom_dock_widget,Qt::Vertical);

    m_slice_setting_widget=new SliceSettingWidget(m_slice_render_widget,this);
    m_volume_render_setting_widget=new VolumeRenderSettingWidget(m_volume_render_widget,this);
    m_volume_setting_widget=new VolumeSettingWidget(this);





    m_setting_dock_widget=new QDockWidget(QStringLiteral("Control Panned"),this);

    m_setting_dock_widget->setAllowedAreas(Qt::LeftDockWidgetArea);
    m_setting_dock_widget->setMinimumSize(400,900);
    m_setting_dock_widget->setMaximumSize(500,1080);
    m_view_menu->addAction(m_setting_dock_widget->toggleViewAction());
    addDockWidget(Qt::LeftDockWidgetArea,m_setting_dock_widget);

    splitDockWidget(m_setting_dock_widget,m_slice_render_dock_widget,Qt::Horizontal);


    m_volume_render_widget->setSlicer(m_slice_render_widget->getSlicer());
    connect(m_slice_render_widget,&SliceRenderWidget::sliceModified,m_volume_render_widget,
            &VolumeRenderWidget::redraw);
}

void VolumeSlicerMainWindow::paintEvent(QPaintEvent* event) {
//    std::cout<<__FUNCTION__ <<std::endl;
//    QPainter p(this);
//
////    multi_renderer->render();
////    auto frame=multi_renderer->GetFrame();
//    Frame frame;
//    frame.width=slicer->GetImageW();
//    frame.height=slicer->GetImageH();
//    frame.channels=1;
//    frame.data.resize((size_t)frame.width*frame.height*frame.channels,0);
////    volume_sampler->Sample(slicer->GetSlice(),frame.data.data());
//    bool complete;
//    START_CPU_TIMER
//    complete=comp_volume_sampler->Sample(slicer->GetSlice(),frame.data.data());
//    END_CPU_TIMER
//    const uchar* data=frame.data.data();
//    QImage image(data,frame.width,frame.height,QImage::Format::Format_Grayscale8,nullptr,nullptr);
////    QImage image(QString(ICONS_PATH)+"open.png");
//
//    auto pix=QPixmap::fromImage(image.mirrored(false,true));
//    auto w=pix.width();
//    p.drawPixmap(0,0,pix);

}
void VolumeSlicerMainWindow::drawVolume() {

}

void VolumeSlicerMainWindow::wheelEvent(QWheelEvent *event) {

    auto angle_delta=event->angleDelta();
    if((QApplication::keyboardModifiers() == Qt::ControlModifier)){
        spdlog::info("{0}",__FUNCTION__ );
        if(angle_delta.y()>0){
            slicer->StretchInXY(1.1f,1.1f);
        }
        else{
            slicer->StretchInXY(0.9f,0.9f);
        }
    }
    else{
        if(angle_delta.y()>0)
            slicer->MoveByNormal(1.f);
        else
            slicer->MoveByNormal(-1.f);
    }
    event->accept();
    repaint();
}
void VolumeSlicerMainWindow::mousePressEvent(QMouseEvent *event) {

    std::cout<<__FUNCTION__<<std::endl;
//    left_mouse_button_pressed=true;
//    last_pos=event->pos();
//    event->accept();
//    repaint();
}
void VolumeSlicerMainWindow::mouseMoveEvent(QMouseEvent *event) {
//    if(left_mouse_button_pressed){
//        auto pos=event->pos();
//        auto d=last_pos-pos;
//        last_pos=pos;
//        std::cout<<d.x()<<" "<<d.y()<<std::endl;
//        slicer->MoveInPlane(d.x(),d.y());
//    }
//    event->accept();
//    repaint();
}

void VolumeSlicerMainWindow::mouseReleaseEvent(QMouseEvent *event) {
    std::cout<<__FUNCTION__<<std::endl;
//    left_mouse_button_pressed=false;
//    event->accept();
//    repaint();
}
void VolumeSlicerMainWindow::keyPressEvent(QKeyEvent *event) {

//    event->accept();
//    repaint();
    std::cout<<__FUNCTION__<<std::endl;

}
void VolumeSlicerMainWindow::initTest() {
//    std::cout<<__FUNCTION__ <<std::endl;
//    raw_volume=RawVolume::Load("C:\\Users\\wyz\\projects\\VolumeSlicer\\test_data\\aneurism_256_256_256_uint8.raw",VoxelType::UInt8,{256,256,256},{0.01f,0.01f,0.01f});
//    multi_renderer=CreateRenderer(1200,900);
//    multi_renderer->SetVolume(raw_volume);
//    Slice slice;
//    slice.origin={9765.f,8434.f,13698.f,1.f};
//    slice.right={1.f,0.f,0.f,0.f};
//    slice.up={0.f,0.f,-1.f,0.f};
//    slice.normal={0.f,1.f,0.f,0.f};
//    slice.n_pixels_width=1200;
//    slice.n_pixels_height=900;
//    slice.voxel_per_pixel_height=2.f;
//    slice.voxel_per_pixel_width=2.f;
//    slicer=Slicer::CreateSlicer(slice);
//    multi_renderer->SetSlicer(slicer);
//    TransferFunc tf;
//    tf.points.emplace_back(0,std::array<double,4>{0.0,0.0,0.0,0.0});
//    tf.points.emplace_back(114,std::array<double,4>{0.5,0.25,0.11,0.0});
//    tf.points.emplace_back(165,std::array<double,4>{0.5,0.25,0.11,0.6});
//    tf.points.emplace_back(216,std::array<double,4>{0.5,0.25,0.11,0.3});
//    tf.points.emplace_back(255,std::array<double,4>{0.0,0.0,0.0,0.0});
//    multi_renderer->SetTransferFunction(std::move(tf));
//    Camera camera;
//    camera.pos={1.28f,1.28f,5.2f};
//    camera.up={0.f,1.f,0.f};
//    camera.front={0.f,0.f,-1.f};
//    camera.zoom=60.f;
//    camera.n=0.01f;
//    camera.f=10.f;
//    multi_renderer->SetCamera(camera);
//    multi_renderer->SetVisible(true,true);
//    volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);


//    this->comp_volume=CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
//    auto block_length=comp_volume->GetBlockLength();
//    std::cout<<"block length: "<<block_length[0]<<" "<<block_length[1]<<std::endl;
//    auto block_dim=comp_volume->GetBlockDim(0);
//    std::cout<<"block dim: "<<block_dim[0]<<" "<<block_dim[1]<<" "<<block_dim[2]<<std::endl;
//
//    comp_volume_sampler=VolumeSampler::CreateVolumeSampler(comp_volume);
//    comp_volume->SetSpaceX(0.01f);
//    comp_volume->SetSpaceY(0.01f);
//    comp_volume->SetSpaceZ(0.03f);
}

