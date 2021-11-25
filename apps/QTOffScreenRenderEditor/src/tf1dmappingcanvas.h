#ifndef TF1DMAPPINGCANVAS_H
#define TF1DMAPPINGCANVAS_H

#include <string>
#include <vector>
#include <QWidget>
#include <QVector2D>

#include "tf1dmappingkey.h"


template <class T> inline
T Clamp(T v, T minv, T maxv)
{
	return (v < minv ? minv : (v > maxv ? maxv : v));
}

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
QT_END_NAMESPACE

class TrivalVolume;

class TF1DMappingCanvas : public QWidget {
	Q_OBJECT

public:
    TF1DMappingCanvas(QWidget * parent = 0, bool noColor = false,
                      bool clipThresholds = false, QString xAxisText = tr("intensity"),
                      QString yAxisText = tr("opacity"));

    virtual ~TF1DMappingCanvas();

    void setTF(float* keysData, int num);

    // Paints the current transfer function in a coordinate system with a grid and a caption.
    virtual void paintEvent(QPaintEvent* event);

    // Coarseness mode is turned on, a new key is inserted when no key is at the mouse position
    // or the context menu is opened when right mousebutton was pressed.
    virtual void mousePressEvent(QMouseEvent* event);

    // Switches coarseness mode off and hides the tooltip.
    virtual void mouseReleaseEvent(QMouseEvent* event);

    // Moves the selected key to new mouse position or moves both keys at the
    // ends of the dragged line. Nothing is done when no key is selected nor a line is dragged.
    virtual void mouseMoveEvent(QMouseEvent* event);

    // Opens a colordialog to change the color of the selected key. Nothing happens when no key is selected.
    virtual void mouseDoubleClickEvent(QMouseEvent* event);

    // Sets the cursor to vertical size cursor when the mouse is on a line of the transfer function and shift was pressed.
    virtual void keyPressEvent(QKeyEvent* event);

    // Unsets the cursor and deletes the selected key when del on keyboard was pressed.
    virtual void keyReleaseEvent(QKeyEvent* event);

    // Creates a default function.
    void createStdFunc();

    // Sets the lower and upper threshold to the given values.
    void setThreshold(float l, float u);

	// Gets the lower and upper threshold to the given values.
    QVector2D getThresholds() const;

    // Returns the minimum size of the widget.
    virtual QSize minimumSizeHint() const;

    // Returns the preferred size of the widget.
    virtual QSize sizeHint() const;

    // Returns the expanding policies of this widget in x and y direction .
    virtual QSizePolicy sizePolicy() const;

    // Sets the caption of the x axis.
    void setXAxisText(const std::string& text);

    // Sets the caption of the y axis.
    void setYAxisText(const std::string& text);

	// Is the thresholds clipped
	bool isThresholdClipped() { return clipThresholds; }

	// Get the 1D transfer function
	void getTransferFunction(float* transferFunction, size_t dimension, float factor);

    // Get the 1D transfer function
    void getTransferFunction(float* transferFunction, int* index, int* num, size_t dimension, float factor);

	// Save the 1D transfer function
	void save(const char* filename);

	// Load the 1D transfer function
	void load(const char* filename);

	void setVolumeInfomation(TrivalVolume * volume) { m_volume = volume; update(); }

signals:
    // Signal that is emitted when the transfer function changed.
    void changed();

    // Signal that is emitted when a transfer function should be loaded from disk.
    void loadTransferFunction();

    // Signal that is emitted when the current transfer function should be saved to disk.
    void saveTransferFunction();

    // Signal that is emitted when the transfer function is reset to default.
    void resetTransferFunction();

    // Signal that is emitted when the user drags a key or a line.
	// It turns coarseness mode on or off.
    void toggleInteraction(bool on);

public slots:
    // Splits or merges the current selected key.
    void splitMergeKeys();

    // Sets the left or right part of the current selected key to zero.
    void zeroKey();

    // Deletes the current selected key.
    void deleteKey();

    // Resets the transfer function to default.
    void resetTransferFunc();

    // Opens a colordialog for choosing the color of the current selected key.
    void changeCurrentColor();

    // Changes the color of the selected key to the given value.
    void changeCurrentColor(const QColor& c);

    // Enables or disables the restriction of visible range to thresholds.
    void toggleClipThresholds();

protected:
    // enum for the status of a key
    enum MarkerProps {
        MARKER_NORMAL   =  0, ///< key is not selected and not split
        MARKER_LEFT     =  1, ///< left part of a key
        MARKER_RIGHT    =  2, ///< right part of a key
        MARKER_SELECTED =  4  ///< key is selected
    };

    // Creates a new key at the given position.
    void insertNewKey(QVector2D& hit);

    // Returns the nearest left or the nearest right key of the given key.
    // If no key exists at left or right 0 is returned.
    TF1DMappingKey* getOtherKey(TF1DMappingKey* selectedKey, bool selectedLeftPart);

    // Returns the number of the key that is left to the mouse position when
    // the position lies on the line between 2 keys. Otherwise -1 is returned.
    int hitLine(const QVector2D& p);

    // Paints all keys of the transfer function.
    void paintKeys(QPainter& paint);

    // Draws the marker at the keys of the transfer function.
    void drawMarker(QPainter& paint, const QColor& color, const QVector2D& p, int props = 0);

	// Draws the histogram
	void drawHistogram(QPainter& paint);

    // Diplays the context menu at the given mouseposition for the case of a keyselection.
    void showKeyContextMenu(QMouseEvent* event);

    // Diplays the context menu at the given mouseposition for the case of no keyselection.
    void showNoKeyContextMenu(QMouseEvent* event);

    // The underlying grid is refined or coarsened according to new size.
    virtual void resizeEvent(QResizeEvent* event);

    // Triggers calculation of the histogram if necessary when canvas is shown.
    virtual void showEvent(QShowEvent* event);

    // Helper function for calculation of pixel coordinates from relative coordinates.
    QVector2D wtos(QVector2D p);

    // Helper function for calculation of relative coordinates from pixel coordinates.
    QVector2D stow(QVector2D p);

    // Hides the tooltip that is displayed when a key is dragged.
    void hideCoordinates();

    // Displays a tooltip at position pos with given values.
    void updateCoordinates(QPoint pos, QVector2D values);

	// Returns the value to which the input value is being mapped.
    // The procedures handles missing keys and out-of-range values gracefully.
    QColor getMappingForValue(float value) const;


protected:
    //ModelData* modelData;	///< model data including the volume, histogram, and transfer function

    float thresholdL;		///< lower threshold in the interval [0, 1] (relative to maximumIntensity_)
    float thresholdU;		///< upper threshold in the interval [0, 1] (relative to maximumIntensity_)

    // variables for interaction
    TF1DMappingKey* selectedKey;	  ///< key that was selected by the user
	std::vector<TF1DMappingKey*> keys;///< internal representation of the transfer function as a set of keys
    bool selectedLeftPart;            ///< when selected key is split, was the left part selected?
    bool dragging;                    ///< is the user dragging a key?
    int dragLine;                     ///< number that indicates the line that was dragged using the shift modifier
    int dragLineStartY;               ///< y position where the drag of the line started
    float dragLineAlphaLeft;          ///< floating alpha value of the left key of the dragged line
    float dragLineAlphaRight;         ///< floating alpha value of the right key of the dragged line
    QPoint mousePos;                  ///< current position of the mouse

    // variables for appearance of widget
    int padding;           ///< additional border of the widget
    int arrowLength;       ///< length of the arrows at the end of coordinate axes
    int arrowWidth;        ///< width of the arrows at the end of coordinate axes
    float splitFactor;     ///< offset between splitted keys
    int pointSize;         ///< size of a key of the transfer function
    int minCellSize;       ///< minimum size of a grid cell
    QVector2D xRange;       ///< range in x direction
    QVector2D yRange;       ///< range in y direction
    QVector2D gridSpacing;  ///< width and height of the underlying grid
    bool clipThresholds;   ///< is the visible range clipped to threshold area?
    bool noColor;          ///< when true the color of a key can not be changed

    QString xAxisText;     ///< caption of the x axis
    QString yAxisText;     ///< caption of the y axis

    QMenu* keyContextMenu;   ///< context menu for right mouse click when a key is selected
    QMenu* noKeyContextMenu; ///< context menu for right mouse click when no key is selected

    QAction* splitMergeAction; ///< action for split/merge context menu entry
    QAction* zeroAction;       ///< action for zero to right context menu entry
    QAction* deleteAction;     ///< action for delete key context menu entry
    QAction* loadAction;       ///< action for load transfer function context menu entry
    QAction* saveAction;       ///< action for save transfer function context menu entry
    QAction* resetAction;      ///< action for reset transfer function context menu entry

	QPixmap* cache;   ///< pixmap for caching the painted histogram

    TrivalVolume * m_volume;
};



#endif
