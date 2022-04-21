//
// Created by wyz on 2021/11/15.
//

#pragma once

#include "camera.hpp"
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QWidget>
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Render/camera.hpp>
using namespace vs;
class CameraVisWidget:public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core{
  public:
    explicit CameraVisWidget(QWidget* parent = nullptr);
    ~CameraVisWidget() override;
    using CameraPoint = decltype(vs::Camera::pos);
    void SetCameraPoints(std::vector<CameraPoint> camera_points);
  public Q_SLOTS:
    void UpdateCameraIndex(int index);

    // set volume board
    void volumeLoaded(const std::shared_ptr<CompVolume>&);

    void volumeClosed();
  private:
    void createCameraIndices();
    void setCameraPoints();
    void normalizeCameraPoints();
    void clear();
  protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width,int height) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent *event) override;
  private:
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo,m_ebo;
    std::unique_ptr<control::Camera> camera;
    std::vector<CameraPoint> camera_points;
    std::vector<uint32_t> indices;
    QOpenGLShaderProgram* m_program;
    int index;
    //draw line
    QOpenGLVertexArrayObject volume_board_vao;
    QOpenGLBuffer volume_board_vbo,volume_board_ebo;

};