//
// Created by csh on 11/17/2021.
//

#include "CameraRouteWidget.h"

namespace{
    const char * cameraVertShader = "#version 330 core \n"
                            "layout(location = 0) in vec3 vPos;\n"
                            "uniform mat4 model;\n"
                            "uniform mat4 view;\n"
                            "uniform mat4 proj;\n"
                            "void main(){\n"
                            "gl_Position = proj*view*model*vec4(vPos,1.0);\n"
                            "}\n";

    const char * cameraFragShader = "#version 330 core\n"
                            "out vec4 fragColor;\n"
                            "void main(){\n"
                            "fragColor = vec4(1.0,0.0,0.0,1.0);\n"
                            "}\n";

    const char * frameVertShader = "#version 330 core \n"
                           "layout(location = 0) in vec3 vPos;\n"
                           "uniform mat4 model;\n"
                           "uniform mat4 view;\n"
                           "uniform mat4 proj;\n"
                           "void main(){\n"
                           "gl_Position = proj*view*model*vec4(vPos,1.0);\n"
                            "}\n";



    const char * frameFragShader = "#version 330 core \n"
                           "out vec4 fragColor;\n"
                            "void main(){\n"
                            "fragColor = vec4(0.6,0.6,0.6,0.5);\n"
                            "}\n";
    const int frameIndices[]={
//        0,1,2,
//        1,2,3,
//        4,5,6,
//        5,6,7,
//        0,1,5,
//        0,4,5,
//        2,3,7,
//        2,6,7,
//        0,2,6,
//        0,4,6,
//        1,5,7,
//        1,3,7
        0,1,2,3,
        4,5,6,7,
        0,1,5,4,
        3,2,6,7,
        0,3,7,4,
        1,2,6,5,
//        0,1,
//        1,2,
//        2,3,
//        3,0,
//        4,5,
//        5,6,
//        6,7,
//        7,4,
//        0,4,
//        1,5,
//        2,6,
//        3,7
    };
}

CameraRouteWidget::CameraRouteWidget(QWidget *parent):QOpenGLWidget(parent),
      x(10.f),
      y(10.f),
      z(10.f),
      volumeLoaded(false),
      frameEBO(QOpenGLBuffer::IndexBuffer),
      frameVBO(QOpenGLBuffer::VertexBuffer),
      routeVBO(QOpenGLBuffer::VertexBuffer)
{
    camera=std::make_unique<control::TrackBallCamera>(
        z/2.f,
        this->width(),this->height(),
        glm::vec3{x/2.f,y/2.f,z/2.f}
//        glm::vec3{0,0,0}
    );
}

void CameraRouteWidget::setXYZ(float in_x, float in_y, float in_z)
{
    makeCurrent();
    x=in_x;
    y=in_y;
    z=in_z;

    camera=std::make_unique<control::TrackBallCamera>(
        z/2.f,
        this->width(),this->height(),
//        glm::vec3{x/2.f,y/2.f,z/2.f},
        glm::vec3{0,0,0}
    );

    float frameVert[]={
        -x/2,y/2,z/2, //top left front
        -x/2,y/2,-z/2, //top left back
        x/2,y/2,-z/2,    //top right back
        x/2,y/2,z/2,   //top right front
        -x/2,-y/2,z/2,
        -x/2,-y/2,-z/2,
        x/2,-y/2,-z/2,
        x/2,-y/2,z/2,
    };

//    float frame[72];
//    for(int i=0;i<sizeof(frameIndices)/sizeof(float);i++){
//        int idx=frameIndices[i];
//        frame[i*3]=frameVert[idx*3];
//        frame[i*3+1]=frameVert[idx*3+1];
//        frame[i*3+2]=frameVert[idx*3+2];
//    }


    if(volumeLoaded){
        frameVBO.destroy();
        frameEBO.destroy();
        frameVAO.destroy();
    }

    if(!routeVAO.create() || !frameEBO.create() || !frameVBO.create()) std::cout<<"create frame object fail"<<std::endl;

    frameVAO.bind();

    frameVBO.bind();
    frameVBO.allocate(frameVert,sizeof(frameVert));

    frameEBO.bind();
    frameEBO.allocate(frameIndices,sizeof(frameIndices));

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glEnableVertexAttribArray(frameVertPos);

    frameVAO.release();
//    frameEBO.release();
    frameVBO.release();

    volumeLoaded = true;
    repaint();
    doneCurrent();
}

//CameraRouteWidget::~CameraRouteWidget(){}

void CameraRouteWidget::setCameraSequence(std::vector<float> &cameras)
{
    makeCurrent();
    camera=std::make_unique<control::TrackBallCamera>(
        z/2.f,
        this->width(),this->height(),
        glm::vec3{x/2.f,y/2.f,z/2.f}
//        glm::vec3{0,0,0}
    );

    if(routeVAO.isCreated())
    {
        routeVAO.destroy();
        routeVBO.destroy();
    }
    cameraSequence.clear();
    cameraSequence.assign(cameras.begin(),cameras.end());

//    float test[]={0.f,0.f,0.f,100.f,100.f,0.0f,0.f,100.f,0.0f};

    routeVAO.create();
    routeVBO.create();
    routeVAO.bind();
    routeVBO.bind();
//    routeVBO.allocate(test,sizeof(test));
    routeVBO.allocate(cameraSequence.data(),cameraSequence.size()*sizeof(float));
    routeShaderProgram->setAttributeBuffer(0,GL_FLOAT,0,3,3*sizeof(float));
    routeShaderProgram->enableAttributeArray(0);
//    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
//    glEnableVertexAttribArray(routeVertPos);
    routeVAO.release();
//    routeVBO.release();
    std::cout<<"cameras size: "<<cameras.size()<<std::endl;

    repaint();
    doneCurrent();
}

void CameraRouteWidget::initializeGL()
{
    makeCurrent();
    initializeOpenGLFunctions();

    routeShaderProgram = new QOpenGLShaderProgram;
    if(!routeShaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,cameraVertShader)){
        qDebug()<<"ERROR:"<<routeShaderProgram->log();
    }
    if(!routeShaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment,cameraFragShader)){
        qDebug()<<"ERROR:"<<routeShaderProgram->log();
    }
    if(!routeShaderProgram->link()){
        qDebug()<<"ERROR:"<<routeShaderProgram->log();
    }
    routeShaderProgram->bind();
    routeVertPos=routeShaderProgram->attributeLocation("vPos");
    routeShaderProgram->release();

    frameShaderProgram = new QOpenGLShaderProgram;
    frameShaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,frameVertShader);
    frameShaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment,frameFragShader);
    frameShaderProgram->link();
    frameShaderProgram->bind();
    frameVertPos=frameShaderProgram->attributeLocation("vPos");
    frameShaderProgram->release();
    doneCurrent();
}

void CameraRouteWidget::resizeGL(int width, int height)
{
    makeCurrent();
    glViewport(0, 0, width, height);
    camera->setScreenSize(width,height);
    doneCurrent();
}

void CameraRouteWidget::paintGL()
{
    makeCurrent();
    QMatrix4x4 model;
    QMatrix4x4 view;
    QMatrix4x4 proj;
    model.setToIdentity();
    view.setToIdentity();
    proj.setToIdentity();

    QVector3D look_at={
        camera->getCameraLookAt().x,
        camera->getCameraLookAt().y,
        camera->getCameraLookAt().z,
    };
    QVector3D up={
        camera->getCameraUp().x,
        camera->getCameraUp().y,
        camera->getCameraUp().z,
    };
    QVector3D pos={
        camera->getCameraPos().x,
        camera->getCameraPos().y,
        camera->getCameraPos().z,
    };
    view.lookAt(pos,look_at,up);
    proj.perspective(camera->getZoom(),float(this->width())/float(this->height()),0.1f,100.f);


    glClearColor(1.0,1.0,1.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT);
//    if(volumeLoaded){
//        QOpenGLVertexArrayObject::Binder binder1(&frameVAO);
//        frameShaderProgram->bind();
//        frameShaderProgram->setUniformValue("model",model);
//        frameShaderProgram->setUniformValue("view",view);
//        frameShaderProgram->setUniformValue("proj",proj);
//        QOpenGLVertexArrayObject::Binder binder0(&frameVAO);
//        glDrawElements(GL_QUADS,sizeof(frameIndices),GL_UNSIGNED_INT,0);
//        frameShaderProgram->release();
//    }

    if(!cameraSequence.empty()){
        QOpenGLVertexArrayObject::Binder binder1(&routeVAO);
        routeShaderProgram->bind();
        routeShaderProgram->setUniformValue("model",model);
        routeShaderProgram->setUniformValue("view",view);
        routeShaderProgram->setUniformValue("proj",proj);
        glDrawArrays(GL_LINE_STRIP,0,cameraSequence.size()/3);
//        glDrawArrays(GL_TRIANGLES,0,3);
        routeShaderProgram->release();
        std::cout <<"paint"<<std::endl;
    }
    doneCurrent();
}

void CameraRouteWidget::mouseMoveEvent(QMouseEvent *event) {
    event->accept();
    if(!camera) return;
    camera->processMouseMove(event->pos().x(),event->pos().y());

    repaint();
}

void CameraRouteWidget::wheelEvent(QWheelEvent *event) {
    event->accept();
    setFocus();
    if(!camera) return;
    camera->processMouseScroll(event->angleDelta().y());

    repaint();
}

void CameraRouteWidget::mousePressEvent(QMouseEvent *event) {
    event->accept();
    setFocus();
    if(!camera) return;
    camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         true,
                                         event->position().x(),
                                         event->position().y());

    repaint();
}

void CameraRouteWidget::mouseReleaseEvent(QMouseEvent *event) {
    event->accept();
    if(!camera) return;
    camera->processMouseButton(control::CameraDefinedMouseButton::Left,
                                         false,
                                         event->position().x(),
                                         event->position().y());

    repaint();
}

void CameraRouteWidget::cleanup(){
    makeCurrent();
    if(!routeShaderProgram){
        delete routeShaderProgram;
        routeShaderProgram=nullptr;
    }

    if(!frameShaderProgram){
        delete frameShaderProgram;
        frameShaderProgram=nullptr;
    }

    routeVAO.destroy();
    routeVBO.destroy();
    frameVAO.destroy();
    frameVBO.destroy();
    frameEBO.destroy();
    volumeLoaded=false;
    cameraSequence.clear();

    doneCurrent();
}