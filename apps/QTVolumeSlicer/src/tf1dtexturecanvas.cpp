
#include "tf1dtexturecanvas.h"
#include "tf1dmappingcanvas.h"
#include <memory>
#include <QDebug>

#define OPENGL_ERROR_MSG							\
{													\
	GLenum error;									\
	while ((error = glGetError()) != GL_NO_ERROR)	\
	{												\
		std::cout<<error;							\
	}												\
}													\

namespace
{
	const char * bgVertShader = "#version 150 \n"
		"in vec3 vPos;\n"
		"in vec3 vColor;\n"
		"out vec3 aColor;\n"
		"uniform mat4 othoMat;\n"
		"void main(){\n"
		"aColor = vColor;\n"
		"gl_Position = othoMat*vec4(vPos,1.0);\n"
		"}\n";

	const char * bgFragShader = "#version 150\n"
		"in vec3 aColor;\n"
		"out vec4 fragColor;\n"
		"void main(){\n"
		"fragColor = vec4(aColor,1.0);\n"
		"}\n";

	const char * tfVertShader = "#version 150 \n"
		"in vec3 vPos;\n"
		"in float vTexCoords;\n"
		"out float aTexCoords;\n"
		"uniform mat4 othoMat;\n"
		"void main(){\n"
		"aTexCoords = vTexCoords;\n"
		"gl_Position = othoMat*vec4(vPos,1.0);\n"
		"}\n";

	const char * tfFragShader = "#version 150\n"
		"in float aTexCoords;\n"
		"uniform sampler1D tfTexture;\n"
		"out vec4 fragColor;\n"
		"void main(){\n"
		"	fragColor = texture(tfTexture,aTexCoords);\n"
		"}\n";


	const float g_bgMesh[] = {
		0       , 0     , -0.5  , 1     , 1     , 1 ,		//vertex and color
		0.5     , 0     , -0.5  , 1     , 1     , 1 ,
		0.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		0       , 0.3   , -0.5  , 1     , 1     , 1 ,
		0.5     , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		1       , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		1       , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		0.5     , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		1       , 0     , -0.5  , 1     , 1     , 1 ,
		1.5     , 0     , -0.5  , 1     , 1     , 1 ,
		1.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		1       , 0.3   , -0.5  , 1     , 1     , 1 ,
		1.5     , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		2       , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		2       , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		1.5     , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		2       , 0     , -0.5  , 1     , 1     , 1 ,
		2.5     , 0     , -0.5  , 1     , 1     , 1 ,
		2.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		2       , 0.3   , -0.5  , 1     , 1     , 1 ,
		2.5     , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		3       , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		3       , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		2.5     , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		3       , 0     , -0.5  , 1     , 1     , 1 ,
		3.5     , 0     , -0.5  , 1     , 1     , 1 ,
		3.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		3       , 0.3   , -0.5  , 1     , 1     , 1 ,
		3.5     , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		4       , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		4       , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		3.5     , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		4       , 0     , -0.5  , 1     , 1     , 1 ,
		4.5     , 0     , -0.5  , 1     , 1     , 1 ,
		4.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		4       , 0.3   , -0.5  , 1     , 1     , 1 ,
		4.5     , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		5       , 0     , -0.5  , 0.6   , 0.6   , 0.6 ,
		5       , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		4.5     , 0.3   , -0.5  , 0.6   , 0.6   , 0.6 ,
		0       , 0.3   , -0.5  , 0.6     , 0.6     , 0.6 ,
		0.5     , 0.3   , -0.5  , 0.6     , 0.6     , 0.6 ,
		0.5     , 0.6   , -0.5  , 0.6     , 0.6     , 0.6 ,
		0       , 0.6   , -0.5  , 0.6     , 0.6     , 0.6 ,
		0.5     , 0.3   , -0.5  ,1     , 1     , 1 ,
		1       , 0.3   , -0.5  ,1     , 1     , 1 ,
		1       , 0.6   , -0.5  ,1     , 1     , 1 ,
		0.5     , 0.6   , -0.5  , 1     , 1     , 1 ,
		1       , 0.3   , -0.5  ,0.6     , 0.6     , 0.6 ,
		1.5     , 0.3   , -0.5  , 0.6     , 0.6     , 0.6 ,
		1.5     , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		1       , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		1.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		2       , 0.3   , -0.5  ,1     , 1     , 1 ,
		2       , 0.6   , -0.5  ,1     , 1     , 1 ,
		1.5     , 0.6   , -0.5  ,1     , 1     , 1 ,
		2       , 0.3   , -0.5  ,0.6     , 0.6     , 0.6 ,
		2.5     , 0.3   , -0.5  , 0.6     , 0.6     , 0.6 ,
		2.5     , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		2       , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		2.5     , 0.3   , -0.5  ,1     , 1     , 1 ,
		3       , 0.3   , -0.5  , 1     , 1     , 1 ,
		3       , 0.6   , -0.5  ,1     , 1     , 1 ,
		2.5     , 0.6   , -0.5  ,1     , 1     , 1 ,
		3       , 0.3   , -0.5  ,0.6     , 0.6     , 0.6 ,
		3.5     , 0.3   , -0.5  ,0.6     , 0.6     , 0.6 ,
		3.5     , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		3       , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		3.5     , 0.3   , -0.5  , 1     , 1     , 1 ,
		4       , 0.3   , -0.5  , 1     , 1     , 1 ,
		4       , 0.6   , -0.5  ,1     , 1     , 1 ,
		3.5     , 0.6   , -0.5  ,1     , 1     , 1 ,
		4       , 0.3   , -0.5  ,0.6     , 0.6     , 0.6 ,
		4.5     , 0.3   , -0.5  ,0.6     , 0.6     , 0.6 ,
		4.5     , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		4       , 0.6   , -0.5  ,0.6     , 0.6     , 0.6 ,
		4.5     , 0.3   , -0.5  ,1     , 1     , 1 ,
		5       , 0.3   , -0.5  , 1     , 1     , 1 ,
		5       , 0.6   , -0.5  ,1     , 1     , 1 ,
		4.5     , 0.6   , -0.5  , 1     , 1     , 1 ,
	};

	const float g_tfVertex[] = {
		0.f,0.f,-0.5f,0.f,					//vertex and 1d texture coords
		5.0f,0.f,-0.5f,1.f,
		5.0f,0.6f,-0.5f,1.f,
		0.f,0.6f,-0.5f,0.f,

	};
}


TF1DTextureCanvas::TF1DTextureCanvas(TF1DMappingCanvas * tf, QWidget *parent)
	: QOpenGLWidget(parent)
	, m_texture(0), m_transferFunction(tf),m_tfShader(nullptr),m_bgShader(nullptr)
{
}
TF1DTextureCanvas::~TF1DTextureCanvas()
{
	cleanup();
	disconnect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &TF1DTextureCanvas::cleanup);
}
void TF1DTextureCanvas::initializeGL()
{

	if (!initializeOpenGLFunctions())
		return;
	glClearColor(1.0, 1.0, 1.0, 1.0);
	connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &TF1DTextureCanvas::cleanup);
	//glBegin();
	//create 1d transfer function texture
	//glEnable(GL_TEXTURE_1D);			//glGetError() == 1280 ?? GL_ERROR_ENUM
	m_tfShader = new QOpenGLShaderProgram;
	m_tfShader->addShaderFromSourceCode(QOpenGLShader::Vertex, tfVertShader);
	m_tfShader->addShaderFromSourceCode(QOpenGLShader::Fragment, tfFragShader);
	m_tfShader->link();
	m_tfShader->bind();
	m_tfVertPos = m_tfShader->attributeLocation("vPos");
	const auto tfTexCoordPos = m_tfShader->attributeLocation("vTexCoords");
	m_tfShader->release();

	m_tfVAO.create();
	m_tfVAO.bind();
	m_tfVBO.create();
	m_tfVBO.bind();
	m_tfVBO.allocate(g_tfVertex, 4 * 4 * sizeof(float));
	glEnableVertexAttribArray(m_tfVertPos);
	glEnableVertexAttribArray(tfTexCoordPos);
	glVertexAttribPointer(m_tfVertPos, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
	glVertexAttribPointer(tfTexCoordPos, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(GLfloat)));

	m_tfVBO.release();
	m_tfVAO.release();

	//Generate texture
	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_1D, m_texture);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glDisable(GL_TEXTURE_1D);

	//Background Shader initialization
	m_bgShader = new QOpenGLShaderProgram;
	m_bgShader->addShaderFromSourceCode(QOpenGLShader::Vertex, bgVertShader);
	m_bgShader->addShaderFromSourceCode(QOpenGLShader::Fragment, bgFragShader);
	m_bgShader->link();
	m_bgShader->bind();
	m_bgVertPos = m_bgShader->attributeLocation("vPos");
	m_bgColorPos = m_bgShader->attributeLocation("vColor");
	m_bgShader->release();

	m_bgVAO.create();
	m_bgVAO.bind();
	m_bgVBO.create();
	m_bgVBO.bind();
	m_bgVBO.allocate(g_bgMesh, sizeof(float) * 20 * 4 * 6);
	glEnableVertexAttribArray(m_bgVertPos);
	glEnableVertexAttribArray(m_bgColorPos);
	glVertexAttribPointer(m_bgVertPos, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
	glVertexAttribPointer(m_bgColorPos, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(GLfloat)));
	m_bgVBO.release();
	m_bgVAO.release();
}

void TF1DTextureCanvas::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
	m_othoMat.setToIdentity();
	m_othoMat.ortho(0, 5, 0, 0.6, 2, -2);
}

void TF1DTextureCanvas::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT);
	QOpenGLVertexArrayObject::Binder binder(&m_bgVAO);
	m_bgShader->bind();
	m_bgShader->setUniformValue("othoMat", m_othoMat);
	glDrawArrays(GL_QUADS, 0, 80);
	m_bgShader->release();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glEnable(GL_TEXTURE_1D);
	//unsigned int texIdx = modelData->getTF1DTextureIdx();
	const int dimension = 256;
	std::unique_ptr<float[]> transferFunction(new float[dimension * 4]);
	m_transferFunction->getTransferFunction(transferFunction.get(), dimension, 1);
	// download 1D Texture Data
	//glEnable(GL_TEXTURE_1D);
	glBindTexture(GL_TEXTURE_1D, m_texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA16, dimension, 0, GL_RGBA, GL_FLOAT, transferFunction.get());
	QOpenGLVertexArrayObject::Binder binder2(&m_tfVAO);
	m_tfShader->bind();
	m_tfShader->setUniformValue("othoMat", m_othoMat);
	glDrawArrays(GL_QUADS, 0, 4);
	m_tfShader->release();
	glDisable(GL_BLEND);
}

void TF1DTextureCanvas::cleanup()
{
	makeCurrent();
	if(m_tfShader != nullptr) {
		delete m_tfShader;
		m_tfShader = nullptr;
	}

	if(m_bgShader != nullptr) {
		delete m_bgShader;
		m_bgShader = nullptr;
	}


	m_bgVAO.destroy();
	m_bgVBO.destroy();
	m_tfVAO.destroy();
	m_tfVBO.destroy();

	if (m_texture != 0) {
		glDeleteTextures(1, &m_texture);
		m_texture = 0;
	}

	doneCurrent();
}
