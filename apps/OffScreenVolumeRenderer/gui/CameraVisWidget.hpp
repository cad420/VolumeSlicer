//
// Created by wyz on 2021/11/15.
//

#pragma once

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include "camera.hpp"
#include <QOpenGLFunctions_3_3_Core>

class CameraVisWidget:public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core{
  public:
    explicit CameraVisWidget(QWidget* parent = nullptr);
    ~CameraVisWidget() override;
    using CameraPoint = std::array<float,3>;
    void SetCameraPoints(std::vector<CameraPoint> camera_points);
  public Q_SLOTS:
    void UpdateCameraIndex(int index);

  private:
    void createCameraIndices();
    void setCameraPoints();
    void normalizeCameraPoints();
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
};