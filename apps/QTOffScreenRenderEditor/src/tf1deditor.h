#ifndef TF1DEDITOR_H
#define TF1DEDITOR_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QCheckBox;
class QComboBox;
class QLayout;
class QSpinBox;
class QToolButton;
QT_END_NAMESPACE

class DoubleSlider;
class TF1DMappingCanvas;
class TF1DTextureCanvas;
class TrivalVolume;
//class GPUVolume;

class TF1DEditor : public QWidget {
	Q_OBJECT

public:
	// Standard constructor
    TF1DEditor(QWidget *parent = 0);
	// Standard destructor
	~TF1DEditor();

	// Widget size hint
	QSize minimumSizeHint() const override;
	QSize sizeHint() const override;

    // Creates the whole layout of this widget.
    void createWidgets();

    // Creates the connections for all control elements.
    void createConnections();

    //set transfer function
    void setTF(float* keys, int num);

	// Get TF1DMapping Canvas
	TF1DMappingCanvas* getTF1DMappingCanvas() { return transCanvas; }

	TF1DTextureCanvas* getTF1DTextureCanvas() { return textureCanvas; }

    // Get the 1D transfer function
    void getTransferFunction(float* transferFunction,int* index, int* num, size_t dimension, float factor);

	// Get the 1D transfer function
	void getTransferFunction(float* transferFunction, size_t dimension, float factor);

	// Set the previous TF file name
	void setTFFileName(QString filename) { curTFFile = filename; }

	// Get the previous TF file name
	QString getTFFileName() { return curTFFile; }

	void setVolumeInformation(TrivalVolume * volume);

    void saveTransferFunctionWithTitle(std::string name);

    void loadTransferFunction(const QString &TFName);


  signals:
    // This signal is emitted when the transfer function has changed.
    void TF1DChanged();

	// Signal that is emitted when the user drags a key or a line.
	// It turns coarseness mode on or off.
    void toggleInteractionMode(bool on);

public slots:
    // Slot for press on reset button. Calls resetThresholds(), resetTransferFunction(),
    // repaints the control elements, notifies the property about the change and emits
    // a repaint for the volume rendering.
	void resetTransferFunction();

    // Slot for a click on load button. Opens a fileDialog and loads the selected
    // file. The gradient widget and the mapping canvas are updated with new transfer function.
    void loadTransferFunction();
//	void loadTransferFunction(const QString &TFName);

    // Slot for a click on save button. Opens a fileDialog and saves the transfer function
    // to the desired file.
    void saveTransferFunction();
//    void saveTransferFunctionWithTitle(std::string name);
	
	// Tells the transfer function that the texture is invalid and emits repaint signal for the
    // volume rendering.
    void updateTransferFunction();

    // Slot that is called from doubleSlider when the user dragged a slider. It updates the
    // spinboxes with new threshold values.
    void thresholdChanged(float min, float max);

    // Slot that is called when the user changes the value in lowerThreshold spinbox.
    // It updates the ranges of spinboxes and adapts the doubleslider to new thresholds.
    // The transfer function is updatet as well.
    void lowerThresholdSpinChanged(int value);

    // Slot that is called when the user changes the value in upperThreshold spinbox.
    // It updates the ranges of spinboxes and adapts the doubleslider to new thresholds.
    // The transfer function is updatet as well.
    void upperThresholdSpinChanged(int value);

    // Starts or stops the interaction mode.
    void toggleInteraction(bool on);

protected:
    // Calls the repaint method for doubleslider, mapping canvas and texture canvas.
    void repaintAll();

    // Applies the current set thresholds to the transfer function.
    void applyThreshold();

    // Resets the threshold to default values. The control elements are updated as well.
    void resetThresholds();

    // Reads the thresholds from the transfer function and assigns them to the widgets.
    void restoreThresholds();

protected:
    TF1DMappingCanvas* transCanvas;       ///< mapping canvas
	TF1DTextureCanvas* textureCanvas;	  ///< canvas that is used for displaying the texture of the transfer function

    QToolButton* loadButton;               ///< button for loading a transfer function
    QToolButton* saveButton;               ///< button for saving a transfer function
    QToolButton* clearButton;              ///< button for resetting transfer function to default
	QToolButton* expandButton;			   ///< button for zooming in threshold area
    QToolButton* repaintButton;            ///< button for forcing a repaint of the volume rendering
    DoubleSlider* doubleSlider;            ///< 2 slider for adjusting the thresholds
    QSpinBox* lowerThresholdSpin;          ///< spinbox for lower threshold
    QSpinBox* upperThresholdSpin;          ///< spinbox for upper threshold

    int maximumIntensity; ///< maximum intensity that can occur in the dataset

	QString curTFFile;

//	GPUVolume * m_volume;

	// Voxel Histogram
	QScopedPointer<double, QScopedPointerArrayDeleter<double>> m_voxelHistogram;
	double m_maxCountOfHistogram;

};

#endif
