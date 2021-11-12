#pragma once


#include <QtOpenGLWidgets/QOpenGLWidget>
//#include <QOpenGLFunctions>

#include <QtOpenGL/QOpenGLShaderProgram>
#include <QtOpenGL/QOpenGLBuffer>
#include <QtOpenGL/QOpenGLVertexArrayObject>
#include <QScopedPointer>

#include <QtOpenGL/QOpenGLFunctions_3_3_Core>

class TF1DMappingCanvas;

class TF1DTextureCanvas : public QOpenGLWidget,protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    TF1DTextureCanvas(TF1DMappingCanvas * tf, QWidget *parent = 0);
    ~TF1DTextureCanvas();

protected:
    void initializeGL() Q_DECL_OVERRIDE;
	void resizeGL(int width, int height)Q_DECL_OVERRIDE;
    void paintGL()Q_DECL_OVERRIDE;

private:
	void cleanup();

    //ModelData *modelData;		///< model data
	TF1DMappingCanvas * m_transferFunction;

	//Background Primitive
	QOpenGLShaderProgram* m_bgShader;
	QOpenGLVertexArrayObject m_bgVAO;
	QOpenGLBuffer m_bgVBO;
	unsigned int m_bgVertPos;
	unsigned int m_bgColorPos;

	//Transfer Function Texture
	QOpenGLShaderProgram* m_tfShader;
	QOpenGLVertexArrayObject m_tfVAO;
	QOpenGLBuffer m_tfVBO;
	unsigned int m_tfVertPos;

	unsigned int m_texture;

	QMatrix4x4 m_othoMat;
					// 2d drawing
};

