//
// Created by wyz on 2021/11/15.
//
#include "CameraVisWidget.hpp"
#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/gl_helper.hpp>
#include <QMatrix4x4>
#include <QWheelEvent>
#include <QMouseEvent>
#include "shaders.hpp"
CameraVisWidget::CameraVisWidget(QWidget *parent)
:QOpenGLWidget(parent)
{

}
void CameraVisWidget::SetCameraPoints(std::vector<CameraPoint> camera_points)
{
    this->camera_points = std::move(camera_points);
    this->index = 0;
    normalizeCameraPoints();

    createCameraIndices();
    setCameraPoints();
}
void CameraVisWidget::UpdateCameraIndex(int index)
{
    this->index = index;
    repaint();
}
void CameraVisWidget::initializeGL()
{
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, [&](){
        clear();
    });

    initializeOpenGLFunctions();
    makeCurrent();
    glClearColor(0,0,0,0);

    camera = std::make_unique<control::TrackBallCamera>(0.5f,width(),height(),glm::vec3{0.5f,0.5f,0.5f});
//    camera = std::make_unique<control::FPSCamera>(glm::vec3{0.5f,0.5f,0.5f});

    m_program = new QOpenGLShaderProgram;
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,shader::line_shader_v);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,shader::line_shader_f);
    if(!m_program->link()){
        LOG_ERROR("shader link failed");
    }
    GL_CHECK
}
void CameraVisWidget::paintGL()
{
    if(camera_points.empty()) return;

    makeCurrent();
    glClearColor(0,0,0,0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    auto view = camera->getViewMatrix();
    auto projection = glm::perspective(glm::radians(camera->getZoom()),float(width())/height(),0.01f,10.f);
    auto mvp = projection * view;
    QMatrix4x4 qmvp(mvp[0][0],mvp[1][0],mvp[2][0],mvp[3][0],
                    mvp[0][1],mvp[1][1],mvp[2][1],mvp[3][1],
                    mvp[0][2],mvp[1][2],mvp[2][2],mvp[3][2],
                    mvp[0][3],mvp[1][3],mvp[2][3],mvp[3][3]);

    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);
    m_program->bind();
    m_program->setUniformValue(m_program->uniformLocation("MVPMatrix"),qmvp);
    m_program->setUniformValue(m_program->uniformLocation("line_color"),1.f,0.f,0.f,1.f);

    glDrawElements(GL_LINES,indices.size(),GL_UNSIGNED_INT,nullptr);

    if(index>0){
        glClear(GL_DEPTH_BUFFER_BIT);
        m_program->setUniformValue(m_program->uniformLocation("line_color"),0.f,1.f,0.f,1.f);
        glDrawElements(GL_LINES,index*2,GL_UNSIGNED_INT,nullptr);
    }

    m_program->release();

}
void CameraVisWidget::resizeGL(int width, int height)
{
    QOpenGLWidget::resizeGL(width, height);
}
void CameraVisWidget::mousePressEvent(QMouseEvent *event)
{
    if(!camera) return;
    camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         true,
                                         event->position().x(),
                                         event->position().y());
    event->accept();
    repaint();
}
void CameraVisWidget::mouseMoveEvent(QMouseEvent *event)
{
    camera->processMouseMove(event->pos().x(),event->pos().y());

    event->accept();
    repaint();
}
void CameraVisWidget::mouseReleaseEvent(QMouseEvent *event)
{
    camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         false,
                                         event->position().x(),
                                         event->position().y());
    event->accept();
    repaint();
}
CameraVisWidget::~CameraVisWidget()
{
    clear();
}

void CameraVisWidget::createCameraIndices()
{
    indices.resize((camera_points.size()-1)*2);
    for(int i =0;i<indices.size()/2;i++){
        indices[2*i] = i;
        indices[2*i+1] = i+1;
    }

}

void CameraVisWidget::setCameraPoints()
{
    makeCurrent();
    m_vao.create();
    m_vao.bind();

    m_vbo = QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
    m_vbo.create();
    m_vbo.bind();
    m_vbo.allocate(camera_points.data(),camera_points.size()*sizeof(CameraPoint));

    m_ebo = QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer);
    m_ebo.create();
    m_ebo.bind();
    m_ebo.allocate(indices.data(),indices.size()*sizeof(uint32_t));
    QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
    f->glEnableVertexAttribArray(0);
    f->glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(GLfloat),nullptr);
    GL_CHECK
}
void CameraVisWidget::normalizeCameraPoints()
{
    float min_x,min_y,min_z,max_x,max_y,max_z;
    min_x = min_y = min_z = std::numeric_limits<float>::max();
    max_x = max_y = max_z = -std::numeric_limits<float>::max();
    for(const auto& pts:camera_points){
        if(pts[0]<min_x) min_x = pts[0];
        if(pts[0]>max_x) max_x = pts[0];
        if(pts[1]<min_y) min_y = pts[1];
        if(pts[1]>max_y) max_y = pts[1];
        if(pts[2]<min_z) min_z = pts[2];
        if(pts[2]>max_z) max_z = pts[2];
    }
    assert(max_x>=min_x && max_y>=min_y && max_z>=min_z);
    auto l = (std::max)({max_x-min_x,max_y-min_y,max_z-min_z});
    for(auto& pts:camera_points){
        pts[0] = (pts[0] - min_x) / l;
        pts[1] = (pts[1] - min_y) / l;
        pts[2] = (pts[2] - min_z) / l;
//        LOG_INFO("normalized pts {0} {1} {2}",pts[0],pts[1],pts[2]);
    }
}
void CameraVisWidget::wheelEvent(QWheelEvent *event)
{
    camera->processMouseScroll(event->angleDelta().y());
    event->accept();
    repaint();
}
void CameraVisWidget::clear()
{
    makeCurrent();
    m_vbo.destroy();
    m_ebo.destroy();
    delete m_program;
    doneCurrent();
}
