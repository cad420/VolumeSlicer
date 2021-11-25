//
// Created by csh on 11/17/2021.
//

#ifndef QTOffScreenRenderEditor_CAMERAROUTEWIDGET_H
#define QTOffScreenRenderEditor_CAMERAROUTEWIDGET_H

#include <QtWidgets>
#include "camera.hpp"
#include <QtOpenGLWidgets>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QtOpenGL/QOpenGLFunctions_3_3_Core>

class CameraRouteWidget:public QOpenGLWidget,protected QOpenGLFunctions_3_3_Core{
    Q_OBJECT
public:
    explicit CameraRouteWidget(QWidget* parent = nullptr);
//    ~CameraRouteWidget();
    void setXYZ(float in_x=10.f, float in_y=10.f, float  in_z=10.f);
    void setCameraSequence(std::vector<float>& cameras);

    void cleanup();
//    void loadVolume();
protected:
//    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
//    void keyPressEvent(QKeyEvent *event) override;

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int width, int height)Q_DECL_OVERRIDE;
    void paintGL()Q_DECL_OVERRIDE;
private:
    float x,y,z;//size of volume
    std::vector<float> cameraSequence;
    std::unique_ptr<control::TrackBallCamera> camera;
//    int w,h;

    QOpenGLShaderProgram* routeShaderProgram;
    QOpenGLBuffer routeVBO;
    QOpenGLVertexArrayObject routeVAO;
    int routeVertPos;

    QOpenGLShaderProgram* frameShaderProgram;
    QOpenGLBuffer frameVBO,frameEBO;
    QOpenGLVertexArrayObject frameVAO;
    int frameVertPos;

    bool volumeLoaded;

};

#endif // QTOffScreenRenderEditor_CAMERAROUTEWIDGET_H
