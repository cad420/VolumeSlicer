//
// Created by wyz on 2021/11/15.
//

#pragma once

#include <QThread>
#include <QWidget>
#include "OffScreenVolumeRenderer.hpp"
#include <thread>
QT_BEGIN_NAMESPACE
class QLineEdit;
class QPushButton;
class QLabel;
class QProgressBar;
QT_END_NAMESPACE


/**
 * @brief emit signal every frame render finish
 */
class RenderProgressWidget: public QWidget{
  Q_OBJECT
  public:
    explicit RenderProgressWidget(QWidget* parent = nullptr);
    ~RenderProgressWidget() override;
    using Pack = std::pair<int,const uint8_t*>;
    void SetRenderConfig(const std::string& config_file,const OffScreenVolumeRenderer::RenderConfig& render_config);
    void render();

  private:
    void createRenderConfigWidgets();

    void updateRenderConfigUI();
    void renderTask();
  Q_SIGNALS:
    void RenderFrameFinish(Pack);
    void RenderStart();
    void RenderStop();

  public Q_SLOTS:
    void renderFrameFinish(Pack);
    void ClearRenderConfigInfo();
    void UpdateRenderProgress(int i,float per);
  private:
    std::string config_file;
    OffScreenVolumeRenderer::RenderConfig render_config;

    QLineEdit* m_input_config_file_le;
    QLineEdit* m_width_le;
    QLineEdit* m_height_le;
    QLineEdit* m_fps_le;
    QLineEdit* m_input_volume_le;
    QLineEdit* m_input_camera_le;
    QLineEdit* m_output_video_le;

    QPushButton* m_start_pb;
    QLabel* m_frame_lb;
    QProgressBar* m_render_progressbar;

    bool exit;

};