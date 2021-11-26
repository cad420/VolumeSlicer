#include <cmath>

#include <QFileDialog>
#include <QLayout>
#include <QSpinBox>
#include <QToolButton>
#include <QDebug>


#include "tf1deditor.h"
#include "tf1dmappingcanvas.h"
#include "DoubleSlider.h"
#include "tf1dtexturecanvas.h"

TF1DEditor::TF1DEditor(QWidget *parent)
	: QWidget(parent)
	, transCanvas(0)
	, textureCanvas(0)
    , doubleSlider(0)
    , maximumIntensity(255)
{
    createWidgets();
    createConnections();
}

TF1DEditor::~TF1DEditor() 
{
}

QSize TF1DEditor::minimumSizeHint() const
{
	return QSize(300, 250);
}

QSize TF1DEditor::sizeHint() const
{
	return QSize(300, 250);
}

void TF1DEditor::createWidgets()
{
	// Buttons and checkBox for threshold clipping
    QHBoxLayout* hboxButton = new QHBoxLayout();
	hboxButton->setSpacing(0);

	clearButton = new QToolButton();
    clearButton->setIcon(QIcon("./icons/TFClear.png"));
    clearButton->setToolTip(tr("Reset to default 1D transfer function"));
	hboxButton->addWidget(clearButton);

    loadButton = new QToolButton();
    loadButton->setIcon(QIcon("./icons/TFOpen.png"));
    loadButton->setToolTip(tr("Load 1D transfer function"));
	hboxButton->addWidget(loadButton);

    saveButton = new QToolButton();
    saveButton->setIcon(QIcon("./icons/TFSave.png"));
    saveButton->setToolTip(tr("Save 1D transfer function"));
    hboxButton->addWidget(saveButton);

	expandButton = new QToolButton();
	expandButton->setIcon(QIcon("./icons/arrowLeftRight.png"));
	expandButton->setToolTip(tr("Zoom-in on the area between lower and upper thresholds"));
	expandButton->setCheckable(true);
	expandButton->setChecked(false);
	hboxButton->addWidget(expandButton);
    hboxButton->addStretch();

	// transfer function mapping area
    transCanvas = new TF1DMappingCanvas(0);
    transCanvas->setMinimumSize(256, 128);

    QWidget* additionalSpace = new QWidget();
    additionalSpace->setMinimumHeight(13);

    // threshold slider
    QHBoxLayout* hboxSlider = new QHBoxLayout();
    doubleSlider = new DoubleSlider();
    doubleSlider->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
    doubleSlider->setOffsets(12, 27);
    hboxSlider->addWidget(doubleSlider);

    //spinboxes for threshold values
    lowerThresholdSpin = new QSpinBox();
    lowerThresholdSpin->setRange(0, maximumIntensity - 1);
    lowerThresholdSpin->setValue(0);
    lowerThresholdSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    upperThresholdSpin = new QSpinBox();
    upperThresholdSpin->setRange(1, maximumIntensity);
    upperThresholdSpin->setValue(maximumIntensity);
    upperThresholdSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    QHBoxLayout* hboxSpin = new QHBoxLayout();
    //the spacing is added so that spinboxes and doubleslider are aligned vertically
    hboxSpin->addSpacing(6);
    hboxSpin->addWidget(lowerThresholdSpin);
    hboxSpin->addStretch();
    hboxSpin->addWidget(upperThresholdSpin);
    hboxSpin->addSpacing(21);

	// texture canvas
    textureCanvas = new TF1DTextureCanvas(transCanvas,this);
    textureCanvas->setFixedHeight(15);
    textureCanvas->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);

    // put widgets in layout
    QVBoxLayout* vBox = new QVBoxLayout();
    //vBox->setMargin(0);
    vBox->setSpacing(1);
    vBox->addStretch();    
	vBox->addLayout(hboxButton);
	vBox->addSpacing(2);
    vBox->addWidget(transCanvas, 1);
    vBox->addWidget(additionalSpace);
	vBox->addWidget(textureCanvas);
	vBox->addSpacing(2);
    vBox->addLayout(hboxSlider);
    vBox->addLayout(hboxSpin);

	setLayout(vBox);
}

void TF1DEditor::createConnections() 
{
    // Buttons
    connect(clearButton, SIGNAL(clicked()), this, SLOT(resetTransferFunction()));
    connect(loadButton, SIGNAL(clicked()), this, SLOT(loadTransferFunction()));
    connect(saveButton, SIGNAL(clicked()), this, SLOT(saveTransferFunction()));
	connect(expandButton, SIGNAL(clicked()), transCanvas, SLOT(toggleClipThresholds()));

    // signals from TF1DMappingCanvas
    connect(transCanvas, SIGNAL(changed()), this, SLOT(updateTransferFunction()));
    connect(transCanvas, SIGNAL(loadTransferFunction()), this, SLOT(loadTransferFunction()));
    connect(transCanvas, SIGNAL(saveTransferFunction()), this, SLOT(saveTransferFunction()));
    connect(transCanvas, SIGNAL(resetTransferFunction()), this, SLOT(resetTransferFunction()));
	connect(transCanvas, SIGNAL(toggleInteraction(bool)), this, SLOT(toggleInteraction(bool)));

    // doubleslider
    connect(doubleSlider, SIGNAL(valuesChanged(float, float)), this, SLOT(thresholdChanged(float, float)));
	connect(doubleSlider, SIGNAL(toggleInteraction(bool)), this, SLOT(toggleInteraction(bool)));

    // threshold spinboxes
    connect(lowerThresholdSpin, SIGNAL(valueChanged(int)), this, SLOT(lowerThresholdSpinChanged(int)));
    connect(upperThresholdSpin, SIGNAL(valueChanged(int)), this, SLOT(upperThresholdSpinChanged(int)));
}

void TF1DEditor::getTransferFunction(float* transferFunction, size_t dimension, float factor)
{
	transCanvas->getTransferFunction(transferFunction, dimension, factor);
	//textureCanvas->update();
	//textureCanvas->repaint();
}

void TF1DEditor::getTransferFunction(float* transferFunction, int* index, int* num, size_t dimension, float factor){
    transCanvas->getTransferFunction(transferFunction, index, num, dimension, factor);
}

void TF1DEditor::setVolumeInformation(TrivalVolume* volume) {
	Q_ASSERT_X(transCanvas, "TF1DEditor::setVolumeInformation", "null pointer");
	transCanvas->setVolumeInfomation(volume);
}

void TF1DEditor::updateTransferFunction()
{
	repaintAll();
    emit TF1DChanged();
}

void TF1DEditor::setTF(float* keys, int num)
{
    transCanvas->setTF(keys, num);
}

void TF1DEditor::loadTransferFunction()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open TF1D"), curTFFile, tr("TF1D (*.TF1D);;All Files (*)"));
	if (!fileName.isEmpty()) {
		transCanvas->load(fileName.toLocal8Bit().data());
		curTFFile = QFileInfo(fileName).absolutePath();
		restoreThresholds();
		repaintAll();
		emit TF1DChanged();
	}
}

void TF1DEditor::loadTransferFunction(const QString & fileName)
{
	if (!fileName.isEmpty()) {
		transCanvas->load(fileName.toLocal8Bit().data());
		curTFFile = QFileInfo(fileName).absolutePath();
		restoreThresholds();
		repaintAll();
		emit TF1DChanged();
	}
}

void TF1DEditor::saveTransferFunction()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save TF1D"), curTFFile, tr("TF1D (*.TF1D);;All Files (*)"));
	if (!fileName.isEmpty()) {
		transCanvas->save(fileName.toLocal8Bit().data());
		curTFFile = QFileInfo(fileName).absolutePath();
	}
}

void TF1DEditor::saveTransferFunctionWithTitle(std::string name)
{
    QString fileName = QString::fromStdString(name + ".TF1D");
//        QFileDialog::getSaveFileName(this, tr("Save TF1D"), curTFFile, tr("TF1D (*.TF1D);;All Files (*)"));
    if (!fileName.isEmpty()) {
        transCanvas->save(fileName.toLocal8Bit().data());
        curTFFile = QFileInfo(fileName).absolutePath();
    }
}

void TF1DEditor::resetTransferFunction() 
{
	resetThresholds();
	transCanvas->createStdFunc();
	repaintAll();
    emit TF1DChanged();
}

void TF1DEditor::resetThresholds()
{
    lowerThresholdSpin->blockSignals(true);
    lowerThresholdSpin->setValue(0);
    lowerThresholdSpin->blockSignals(false);

    upperThresholdSpin->blockSignals(true);
    upperThresholdSpin->setValue(maximumIntensity);
    upperThresholdSpin->blockSignals(false);

    doubleSlider->blockSignals(true);
    doubleSlider->setValues(0.f, 1.f);
    doubleSlider->blockSignals(false);

    transCanvas->setThreshold(0.f, 1.f);
}

void TF1DEditor::thresholdChanged(float min, float max) 
{
    //convert to integer values
    int val_min = static_cast<int>(std::floor(min * maximumIntensity + 0.5));
    int val_max = static_cast<int>(std::floor(max * maximumIntensity + 0.5));

    //sync with spinboxes
    if ((val_max != upperThresholdSpin->value()))
        upperThresholdSpin->setValue(val_max);

    if ((val_min != lowerThresholdSpin->value()))
        lowerThresholdSpin->setValue(val_min);

    //apply threshold to transfer function
    applyThreshold();
}

void TF1DEditor::lowerThresholdSpinChanged(int value) 
{
    if (value+1 < maximumIntensity) {
        //increment maximum of lower spin when maximum was reached and we are below upper range
        if (value == lowerThresholdSpin->maximum())
            lowerThresholdSpin->setMaximum(value+1);

        //update minimum of upper spin
        upperThresholdSpin->blockSignals(true);
        upperThresholdSpin->setMinimum(value);
        upperThresholdSpin->blockSignals(false);
    }
    //increment value of upper spin when it equals value of lower spin
    if (value == upperThresholdSpin->value()) {
        upperThresholdSpin->blockSignals(true);
        upperThresholdSpin->setValue(value+1);
        upperThresholdSpin->blockSignals(false);
    }

    //update doubleSlider to new minValue
    doubleSlider->blockSignals(true);
    doubleSlider->setMinValue(value / static_cast<float>(maximumIntensity));
    doubleSlider->blockSignals(false);

    //apply threshold to transfer function
    applyThreshold();
}

void TF1DEditor::upperThresholdSpinChanged(int value) 
{
    if (value-1 > 0) {
        //increment minimum of upper spin when minimum was reached and we are above lower range
        if (value == upperThresholdSpin->minimum())
            upperThresholdSpin->setMinimum(value-1);

        //update maximum of lower spin
        lowerThresholdSpin->blockSignals(true);
        lowerThresholdSpin->setMaximum(value);
        lowerThresholdSpin->blockSignals(false);
    }
    //increment value of lower spin when it equals value of upper spin
    if (value == lowerThresholdSpin->value()) {
        lowerThresholdSpin->blockSignals(true);
        lowerThresholdSpin->setValue(value-1);
        lowerThresholdSpin->blockSignals(false);
    }

    //update doubleSlider to new maxValue
    doubleSlider->blockSignals(true);
    doubleSlider->setMaxValue(value / static_cast<float>(maximumIntensity));
    doubleSlider->blockSignals(false);

    //apply threshold to transfer function
    applyThreshold();
}

void TF1DEditor::toggleInteraction(bool on)
{
	emit toggleInteractionMode(on);	
}

void TF1DEditor::applyThreshold()
{
    float min = doubleSlider->getMinValue();
    float max = doubleSlider->getMaxValue();

    transCanvas->setThreshold(min, max);
	///////
	// /
	textureCanvas->update();
	///////
    updateTransferFunction();
}

void TF1DEditor::restoreThresholds() 
{
	QVector2D thresh = transCanvas->getThresholds();

    // set value for doubleSlider
    doubleSlider->blockSignals(true);
    doubleSlider->setValues(thresh.x(), thresh.y());
    doubleSlider->blockSignals(false);

    // set value for spinboxes
    int val_min = static_cast<int>(std::floor(thresh.x() * maximumIntensity + 0.5));
    int val_max = static_cast<int>(std::floor(thresh.y() * maximumIntensity + 0.5));
    lowerThresholdSpin->blockSignals(true);
    upperThresholdSpin->blockSignals(true);
    lowerThresholdSpin->setValue(val_min);
    upperThresholdSpin->setValue(val_max);
    lowerThresholdSpin->blockSignals(false);
    upperThresholdSpin->blockSignals(false);
    // test whether to update minimum and/or maximum of spinboxes
    if (val_min+1 < maximumIntensity) {
        //increment maximum of lower spin when maximum was reached and we are below upper range
        if (val_min == lowerThresholdSpin->maximum())
            lowerThresholdSpin->setMaximum(val_min+1);

        //update minimum of upper spin
        upperThresholdSpin->blockSignals(true);
        upperThresholdSpin->setMinimum(val_min);
        upperThresholdSpin->blockSignals(false);
    }

    if (val_max-1 > 0) {
        //increment minimum of upper spin when minimum was reached and we are above lower range
        if (val_max == upperThresholdSpin->minimum())
            upperThresholdSpin->setMinimum(val_max-1);

        //update maximum of lower spin
        lowerThresholdSpin->blockSignals(true);
        lowerThresholdSpin->setMaximum(val_max);
        lowerThresholdSpin->blockSignals(false);
    }

    // propagate threshold to mapping canvas
    transCanvas->setThreshold(thresh.x(), thresh.y());
}

void TF1DEditor::repaintAll() 
{
    transCanvas->update();
    doubleSlider->update();
    textureCanvas->update();
}
