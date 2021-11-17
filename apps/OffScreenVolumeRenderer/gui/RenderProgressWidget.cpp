//
// Created by wyz on 2021/11/15.
//
#include "RenderProgressWidget.hpp"
#include <QLineEdit>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QtConcurrent/QtConcurrent>
#include <VolumeSlicer/Utils/logger.hpp>
RenderProgressWidget::RenderProgressWidget(QWidget *parent)
:QWidget(parent)
{
    createRenderConfigWidgets();
}

void RenderProgressWidget::createRenderConfigWidgets()
{
    auto layout = new QVBoxLayout();

    auto input_file_layout = new QHBoxLayout();
    auto input_file_label = new QLabel("Input Config File");
    this->m_input_config_file_le = new QLineEdit();
    this->m_input_config_file_le->setReadOnly(true);
    auto output_video_label = new QLabel("Output Video");
    this->m_output_video_le = new QLineEdit();
    this->m_output_video_le->setReadOnly(true);
    input_file_layout->addWidget(input_file_label);
    input_file_layout->addWidget(m_input_config_file_le);
    input_file_layout->addWidget(output_video_label);
    input_file_layout->addWidget(m_output_video_le);
    layout->addLayout(input_file_layout);

    auto frame_info_layout = new QHBoxLayout();
    auto frame_width_label = new QLabel("Width");
    this->m_width_le = new QLineEdit();
    this->m_width_le->setReadOnly(true);
    auto frame_height_label = new QLabel("Height");
    this->m_height_le = new QLineEdit();
    this->m_height_le->setReadOnly(true);
    auto frame_fps_label = new QLabel("FPS");
    this->m_fps_le = new QLineEdit();
    this->m_fps_le->setReadOnly(true);
    frame_info_layout->addWidget(frame_width_label);
    frame_info_layout->addWidget(m_width_le);
    frame_info_layout->addWidget(frame_height_label);
    frame_info_layout->addWidget(m_height_le);
    frame_info_layout->addWidget(frame_fps_label);
    frame_info_layout->addWidget(m_fps_le);
    layout->addLayout(frame_info_layout);

    auto input_info_layout = new QHBoxLayout();
    auto input_volume_label = new QLabel("Input Volume");
    this->m_input_volume_le = new QLineEdit();
    this->m_input_volume_le->setReadOnly(true);
    auto input_camera_label = new QLabel("Input Camera");
    this->m_input_camera_le = new QLineEdit();
    this->m_input_camera_le->setReadOnly(true);
    input_info_layout->addWidget(input_volume_label);
    input_info_layout->addWidget(m_input_volume_le);
    input_info_layout->addWidget(input_camera_label);
    input_info_layout->addWidget(m_input_camera_le);

    layout->addLayout(input_info_layout);

    auto start_render_layout = new QHBoxLayout();
    this->m_start_pb = new QPushButton("Start");
    this->m_start_pb->setEnabled(false);
    this->m_frame_lb = new QLabel("Frame");
    this->m_render_progressbar = new QProgressBar();
    m_render_progressbar->setValue(0);
    start_render_layout->addWidget(m_start_pb);
    start_render_layout->addWidget(m_frame_lb);
    start_render_layout->addWidget(m_render_progressbar);
    layout->addLayout(start_render_layout);

    this->setLayout(layout);

    connect(m_start_pb,&QPushButton::clicked,this,[&](){
        render();
    });
    connect(this,&RenderProgressWidget::RenderStart,this,[&](){
        m_start_pb->setEnabled(false);
    });

    connect(this,&RenderProgressWidget::RenderStop,this,[&](){
        m_start_pb->setEnabled(true);
    });
    connect(m_render_progressbar,&QProgressBar::valueChanged,this->m_render_progressbar,
            &QProgressBar::setValue);
}

void RenderProgressWidget::updateRenderConfigUI()
{
    m_input_config_file_le->setText(config_file.c_str());
    m_width_le->setText(std::to_string(render_config.width).c_str());
    m_height_le->setText(std::to_string(render_config.height).c_str());
    m_fps_le->setText(std::to_string(render_config.fps).c_str());
    m_input_volume_le->setText(render_config.volume_data_config_file.c_str());
    m_input_camera_le->setText(render_config.camera_sequence_config.c_str());
    m_output_video_le->setText(render_config.output_video_name.c_str());
    m_start_pb->setEnabled(true);
}

void RenderProgressWidget::SetRenderConfig(const std::string& config_file,const OffScreenVolumeRenderer::RenderConfig &render_config)
{
    this->config_file = config_file;
    this->render_config = render_config;
    updateRenderConfigUI();
}

void RenderProgressWidget::render()
{
    exit = false;
    QtConcurrent::run(&RenderProgressWidget::renderTask,this);
}
void RenderProgressWidget::ClearRenderConfigInfo()
{
    m_input_config_file_le->clear();
    m_width_le->clear();
    m_height_le->clear();
    m_fps_le->clear();
    m_input_volume_le->clear();
    m_input_camera_le->clear();
    m_output_video_le->clear();
}
void RenderProgressWidget::UpdateRenderProgress(int i,float per)
{
    std::string str = "Frame "+std::to_string(i+1);
    m_frame_lb->setText(str.c_str());
//    m_render_progressbar->setValue(per*100);
    emit m_render_progressbar->valueChanged(per*100);
}
RenderProgressWidget::~RenderProgressWidget()
{
    exit = true;
}
void RenderProgressWidget::renderFrameFinish(RenderProgressWidget::Pack pack)
{
    emit RenderFrameFinish(pack);
}
void RenderProgressWidget::renderTask()
{
    //todo data may be invalid when access by next because emit wouldn't block callback
    OffScreenVolumeRenderer::Callback callback=[this](int frame,float per,const uint8_t* data){
      UpdateRenderProgress(frame,per);
      Pack pack{frame,data};
      LOG_INFO("In callback before emit");
      renderFrameFinish(pack);
      LOG_INFO("In callback after emit");
      if(exit){
          QCoreApplication::exit(0);
      }
    };

    emit RenderStart();
    OffScreenVolumeRenderer::RenderFrames(render_config,callback);
    emit RenderStop();
}
