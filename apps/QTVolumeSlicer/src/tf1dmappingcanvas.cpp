#include <algorithm>
#include <sstream>
#include <QMenu>
#include <QAction>
#include <QColor>
#include <QColorDialog>
#include <QMouseEvent>
#include <QPainter>
#include <QToolTip>
#include <QMessageBox>
#include <cmath>
#include <QMatrix4x4>

#include "tf1dmappingcanvas.h"

#include "TrivalVolume.hpp"

bool sortFunction(TF1DMappingKey* a, TF1DMappingKey* b) 
{
    return a->getIntensity() < b->getIntensity();
}

TF1DMappingCanvas::TF1DMappingCanvas(QWidget* parent, bool noColor_,
                                     bool clipThresholds_, QString xAxisText_, QString yAxisText_)
    : QWidget(parent)
    , clipThresholds(clipThresholds_)
    , noColor(noColor_)
    , xAxisText(xAxisText_)
    , yAxisText(yAxisText_)
	, cache(0)
	, m_volume(0)
{
    xRange = QVector2D(0.f, 1.f);
    yRange = QVector2D(0.f, 1.f);
    padding = 12;
    arrowLength = 10;
    arrowWidth = 3;
    pointSize = 10;
    selectedKey = 0;
    selectedLeftPart = true;
    splitFactor = 1.5f;
    minCellSize = 8;
    dragging = false;
    dragLine = -1;
    dragLineAlphaLeft = -1.f;
    dragLineAlphaRight = -1.f;

    setObjectName("TF1DMappingCanvas");
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);

    setFocus();

	createStdFunc();

    setThreshold(0.f, 1.f);	
	
	keyContextMenu = new QMenu();
	noKeyContextMenu = new QMenu();

    if (!noColor) {
        QAction* cc = new QAction(tr("Change color of key"), this);
        keyContextMenu->addAction(cc);
        connect(cc, SIGNAL(triggered()), this, SLOT(changeCurrentColor()));
    }

    splitMergeAction = new QAction(tr(""), this); // Text will be set later
    keyContextMenu->addAction(splitMergeAction);
    connect(splitMergeAction, SIGNAL(triggered()), this, SLOT(splitMergeKeys()));

    zeroAction = new QAction("", this); // Text will be set later
    keyContextMenu->addAction(zeroAction);
    connect(zeroAction, SIGNAL(triggered()), this, SLOT(zeroKey()));

    deleteAction = new QAction(tr("Delete this key"), this);
    keyContextMenu->addAction(deleteAction);
    connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteKey()));

    loadAction = new QAction(tr("Load transfer function..."), this);
    noKeyContextMenu->addAction(loadAction);
    connect(loadAction, SIGNAL(triggered()), this, SIGNAL(loadTransferFunction()));

    saveAction = new QAction(tr("Save transfer function..."), this);
    noKeyContextMenu->addAction(saveAction);
    connect(saveAction, SIGNAL(triggered()), this, SIGNAL(saveTransferFunction()));

    resetAction = new QAction(tr("Reset transfer function"), this);
    noKeyContextMenu->addAction(resetAction);
    connect(resetAction, SIGNAL(triggered()), this, SLOT(resetTransferFunc()));
}

TF1DMappingCanvas::~TF1DMappingCanvas() 
{
	for(size_t i = 0; i < keys.size(); ++i)
		delete keys[i];
	keys.clear();
}

//--------- methods for reacting on Qt events ---------//

void TF1DMappingCanvas::showNoKeyContextMenu(QMouseEvent *event) 
{
    noKeyContextMenu->popup(event->globalPos());
}

void TF1DMappingCanvas::resizeEvent(QResizeEvent* event) 
{
    QWidget::resizeEvent(event);
    gridSpacing = QVector2D(1.0, 1.0);
    // refine gridSpacing_ as good as possible
    QVector2D factor = QVector2D(0.1f, 0.2f);
    for (int k=0; k<2; ++k) {
        for (int component=0; component<2; ++component) {
            QVector2D cellSize = wtos(gridSpacing) - wtos(QVector2D(0.0, 0.0));
            cellSize[component] *= factor[k];
            while (cellSize[component] > minCellSize) {
                gridSpacing[component] *= factor[k];
                cellSize[component] *= factor[k];
            }
            cellSize[component] /= factor[k];
        }
    }
}

void TF1DMappingCanvas::showEvent(QShowEvent* event) 
{
    QWidget::showEvent(event);
    //updateHistogram(); // only if necessary
}

void TF1DMappingCanvas::showKeyContextMenu(QMouseEvent* event) 
{
    // Set context-dependent text for menu items

    // Split/merge
    QString splitMergeText;
    if (selectedKey->isSplit())
        splitMergeText = tr("Merge this key");
    else
        splitMergeText = tr("Split this key");
    splitMergeAction->setText(splitMergeText);

    // Zero/unzero
    QString zeroText;
    if (selectedLeftPart)
        zeroText = tr("Zero to the left");
    else
        zeroText = tr("Zero to the right");
    zeroAction->setText(zeroText);

    // allow deletion of keys only if there are more than two keys
    deleteAction->setEnabled(keys.size() > 2);

    keyContextMenu->popup(event->globalPos());
}

void TF1DMappingCanvas::paintEvent(QPaintEvent* event)
{
    event->accept();

    QPainter paint(this);

    // put origin in lower lefthand corner
//    QMatrix4x4 m;
    QTransform m;
    m.translate(0.0, static_cast<float>(height())-1);
    m.scale(1.f, -1.f);
    paint.setTransform(m);
    paint.setWorldMatrixEnabled(true);

    paint.setRenderHint(QPainter::Antialiasing, false);
    paint.setPen(Qt::NoPen);
    paint.setBrush(Qt::white);
    paint.drawRect(0, 0, width() - 1, height() - 1);

    // ----------------------------------------------

    // draw grid
    paint.setPen(QColor(220, 220, 220));
    paint.setRenderHint(QPainter::Antialiasing, false);

    QVector2D pmin = QVector2D(0.f, 0.f);
    QVector2D pmax = QVector2D(1.f, 1.f);

    for (float f=pmin.x(); f<pmax.x()+gridSpacing.x()*0.5; f+=gridSpacing.x()) {
        QVector2D p = wtos(QVector2D(f, 0.f));
        QVector2D a = wtos(QVector2D(0.f, 0.f));
        QVector2D b = wtos(QVector2D(0.f, 1.f));
        paint.drawLine(QPointF(p.x(), a.y()),
                       QPointF(p.x(), b.y()));
    }

    for (float f=pmin.y(); f<pmax.y()+gridSpacing.y()*0.5; f+=gridSpacing.y()) {
        QVector2D p = wtos(QVector2D(0.f, f));
        QVector2D a = wtos(QVector2D(0.f, 0.f));
        QVector2D b = wtos(QVector2D(1.f, 0.f));
        paint.drawLine(QPointF(a.x(), p.y()),
                       QPointF(b.x(), p.y()));
    }

    // draw x and y axes
    paint.setRenderHint(QPainter::Antialiasing, true);
    paint.setPen(Qt::gray);
    paint.setBrush(Qt::gray);

    // draw axes independently from visible range
    float oldx0 = xRange[0];
    float oldx1 = xRange[1];
    xRange[0] = 0.f;
    xRange[1] = 1.f;

    QVector2D origin = wtos(QVector2D(0.f, 0.f));
    origin.setX(std::floor(origin.x()) + 0.5f);
    origin.setY(std::floor(origin.y()) + 0.5f);

    paint.setRenderHint(QPainter::Antialiasing, true);

    paint.drawLine(QPointF(padding, origin.y()),
                   QPointF(width() - padding, origin.y()));

    paint.drawLine(QPointF(origin.x(), padding),
                   QPointF(origin.x(), height() - padding));

    QPointF arrow[3];
    arrow[0] = QPointF(origin.x(), height() - padding);
    arrow[1] = QPointF(origin.x() + arrowWidth, height() - padding - arrowLength);
    arrow[2] = QPointF(origin.x() - arrowWidth, height() - padding - arrowLength);

    paint.drawConvexPolygon(arrow, 3);

    arrow[0] = QPointF(width() - padding, origin.y());
    arrow[1] = QPointF(width() - padding - arrowLength, origin.y() - arrowWidth);
    arrow[2] = QPointF(width() - padding - arrowLength, origin.y() + arrowWidth);

    paint.drawConvexPolygon(arrow, 3);

    paint.scale(-1.f, 1.f);
    paint.rotate(180.f);
    paint.drawText(static_cast<int>(width() - 6.2f * padding), static_cast<int>(-1 * (origin.y() - 0.8f * padding)), xAxisText);
    paint.drawText(static_cast<int>(1.6f * padding), static_cast<int>(-1 * (height() - 1.85f * padding)), yAxisText);

    paint.rotate(180.f);
    paint.scale(-1.f, 1.f);

    xRange[0] = oldx0;
    xRange[1] = oldx1;

    // ----------------------------------------------

    // draw mapping function
    QPen pen = QPen(Qt::darkRed);
    pen.setWidthF(1.5f);
    paint.setPen(pen);

    origin = wtos(QVector2D(0.f,0.f));

    QVector2D old;
	for (size_t i=0; i<keys.size(); ++i) {
        TF1DMappingKey *key = keys[i];
        QVector2D p = wtos(QVector2D(key->getIntensity(), key->getColorL().alpha() / 255.f));
        if (i == 0)  {
            if (keys[0]->getIntensity() > 0.f)
                paint.drawLine(QPointF(wtos(QVector2D(0.f, 0.f)).x(), p.y()),
                               QPointF(p.x() - 1.f, p.y()));
        }
        else {
            paint.drawLine(QPointF(old.x() + 1.f, old.y()),
                           QPointF(p.x() - 1.f, p.y()));
        }
        old = p;
        if (key->isSplit())
            old = wtos(QVector2D(key->getIntensity(), key->getColorR().alpha() / 255.f));
    }
    if (keys[keys.size() - 1]->getIntensity() < 1.f) {
        paint.drawLine(QPointF(old.x() + 1.f, old.y()),
                       QPointF(wtos(QVector2D(1.f, 0.f)).x(), old.y()));
    }

    if (xRange[1] != xRange[0])
        paintKeys(paint);

    // ----------------------------------------------

    // grey out threshold area
    paint.setBrush(QBrush(QColor(192, 192, 192, 230), Qt::SolidPattern));
    paint.setPen(Qt::NoPen);
    QVector2D upperRight = wtos(QVector2D(1.f, 1.f));
    QVector2D lowerLeft = wtos(QVector2D(0.f, 0.f));
    int w = static_cast<int>(upperRight.x() - lowerLeft.x());
    int h = static_cast<int>(upperRight.y() - lowerLeft.y());

    if (thresholdL > 0.f) {
        paint.drawRect(static_cast<int>(origin.x()), static_cast<int>(origin.y()),
                       static_cast<int>(thresholdL * w + 1), h);
    }
    if (thresholdU < 1.f) {
        paint.drawRect(static_cast<int>(origin.x() + floor(thresholdU * w)),
                       static_cast<int>(origin.y()), static_cast<int>((1 - thresholdU) * w + 1), h);
    }

    paint.setRenderHint(QPainter::Antialiasing, false);

    paint.setPen(Qt::lightGray);
    paint.setBrush(Qt::NoBrush);
    paint.drawRect(0, 0, width() - 1, height() - 1);

    paint.setWorldMatrixEnabled(false);

	// ----------------------------------------------
	drawHistogram(paint);
}

void TF1DMappingCanvas::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton)
		emit toggleInteraction(true);

    event->accept();

    dragLine = hitLine(QVector2D(event->x(), event->y()));
    if (dragLine >= 0 && event->modifiers() == Qt::ShiftModifier) {
        dragLineStartY = event->y();
        return;
    }

    QVector2D sHit = QVector2D(event->x(), static_cast<float>(height()) - event->y());
    QVector2D hit = stow(sHit);

    // see if a key was selected
    selectedKey = 0;
	for (size_t i=0; i<keys.size(); ++i) {
        TF1DMappingKey* key = keys[i];
        QVector2D sp = wtos(QVector2D(key->getIntensity(), key->getColorL().alpha() / 255.0));
        QVector2D spr = wtos(QVector2D(key->getIntensity(), key->getColorR().alpha() / 255.0));
        if (key->isSplit()) {
            if (sHit.x() > sp.x() - splitFactor * pointSize && sHit.x() <= sp.x() &&
                sHit.y() > sp.y() - pointSize && sHit.y() < sp.y() + pointSize) {
                selectedKey = key;
                selectedLeftPart = true;
            }
            if (sHit.x() >= spr.x() && sHit.x() < spr.x() + splitFactor * pointSize &&
                sHit.y() > spr.y() - pointSize && sHit.y() < spr.y() + pointSize) {
                selectedKey = key;
                selectedLeftPart = false;
            }
        }
        else {
            if (sHit.x() > sp.x() - pointSize && sHit.x() < sp.x() + pointSize &&
                sHit.y() > sp.y() - pointSize && sHit.y() < sp.y() + pointSize) {
                selectedKey = key;
                selectedLeftPart = false;
            }
        }
    }


    if (event->button() == Qt::RightButton) {
        if (selectedKey == 0)
            showNoKeyContextMenu(event);
        else
            showKeyContextMenu(event);
        return;
    }

    if (selectedKey != 0 && event->button() == Qt::LeftButton) {
        dragging = true;
        //keep values within valid range
        hit.setX(Clamp(hit.x(), 0.f, 1.f));
		hit.setY(Clamp(hit.y(), 0.f, 1.f));
        updateCoordinates(event->pos(), hit);
        return;
    }

    // no key was selected -> insert new key
    if (hit.x() >= 0.f && hit.x() <= 1.f &&
        hit.y() >= 0.f && hit.y() <= 1.f &&
        event->button() == Qt::LeftButton)
    {
        insertNewKey(hit);
        dragging = true;
        dragLine = -1;
        updateCoordinates(event->pos(), hit);
        update();
        emit changed();
    }
}

void TF1DMappingCanvas::mouseMoveEvent(QMouseEvent* event) 
{
    event->accept();
    mousePos = event->pos();

    QVector2D sHit = QVector2D(event->x(), static_cast<float>(height()) - event->y());
    QVector2D hit = stow(sHit);

    if (!dragging && hitLine(QVector2D(event->x(), event->y())) >= 0 && event->modifiers() == Qt::ShiftModifier)
        setCursor(Qt::SizeVerCursor);
    else
        unsetCursor();

    if (dragLine >= 0) {
        // a line between 2 keys is moved (shift modifier was used)
        float delta = dragLineStartY - event->y();
        dragLineStartY = event->y();
        //left key
        TF1DMappingKey* key = keys[dragLine];
        if (dragLineAlphaLeft == -1.f)
            dragLineAlphaLeft = key->isSplit() ? key->getAlphaR() : key->getAlphaL();
        dragLineAlphaLeft = wtos(QVector2D(dragLineAlphaLeft,0.f)).y();
        dragLineAlphaLeft += delta;
        dragLineAlphaLeft = stow(QVector2D(dragLineAlphaLeft,0.f)).y();
        if (dragLineAlphaLeft < 0.f)
            dragLineAlphaLeft = 0.f;
        if (dragLineAlphaLeft > 1.f)
            dragLineAlphaLeft = 1.f;
        key->setAlphaR(dragLineAlphaLeft);
		std::sort(keys.begin(), keys.end(), [](const TF1DMappingKey * a, const TF1DMappingKey * b){ return a->getIntensity() < b->getIntensity(); });
        if (keys.size() > dragLine+1) {
            //right key - when existing
            key = keys[dragLine+1];
            if (dragLineAlphaRight == -1.f)
                dragLineAlphaRight = key->getAlphaL();
            dragLineAlphaRight = wtos(QVector2D(dragLineAlphaRight,0.f)).y();
            dragLineAlphaRight += delta;
            dragLineAlphaRight = stow(QVector2D(dragLineAlphaRight,0.f)).y();
            if (dragLineAlphaRight < 0.f)
                dragLineAlphaRight = 0.f;
            if (dragLineAlphaRight > 1.f)
                dragLineAlphaRight = 1.f;
            key->setAlphaL(dragLineAlphaRight);
			std::sort(keys.begin(), keys.end(), [](const TF1DMappingKey * a, const TF1DMappingKey * b) { return a->getIntensity() < b->getIntensity(); });
        }
        update();
        emit changed();
        return;
    }

    // return when no key was inserted or selected
    if (!dragging)
        return;

    // keep location within valid texture coord range
    hit.setX(Clamp(hit.x(), 0.f, 1.f));
	hit.setY(Clamp(hit.y(), 0.f, 1.f));

    if (selectedKey != 0) {
        updateCoordinates(event->pos(), hit);
        if (event->modifiers() != Qt::ShiftModifier) {
            selectedKey->setIntensity(hit.x());
        }
        if (event->modifiers() != Qt::ControlModifier) {
            if (selectedKey->isSplit()) {
                if (selectedLeftPart)
                    selectedKey->setAlphaL(hit.y());
                else
                    selectedKey->setAlphaR(hit.y());
            }
            else
                selectedKey->setAlphaL(hit.y());
        }
        bool selectedFound = false;
        for (size_t i = 0; i < keys.size(); ++i) {
            TF1DMappingKey* key = keys[i];
            //is the tf key the selected one?
            if (key == selectedKey) {
                selectedFound = true;
                continue;
            }
            if (selectedFound) {
                //change intensity of key if its lower than the intensity of selectedKey
                if (key->getIntensity() < selectedKey->getIntensity())
                    key->setIntensity(selectedKey->getIntensity());
            }
            else {
                //change intensity of key if its higher than the intensity of selectedKey
                if (key->getIntensity() > selectedKey->getIntensity())
                    key->setIntensity(selectedKey->getIntensity());
            }
        }
		std::sort(keys.begin(), keys.end(), [](const TF1DMappingKey * a, const TF1DMappingKey * b) { return a->getIntensity() < b->getIntensity(); });
        update();
        emit changed();
    }
}

void TF1DMappingCanvas::mouseReleaseEvent(QMouseEvent* event)
{
    event->accept();
    if (event->button() == Qt::LeftButton) {
        dragging = false;
        dragLine = -1;
        dragLineAlphaLeft = -1.f;
        dragLineAlphaRight = -1.f;
        hideCoordinates();
        update();
		emit toggleInteraction(false);
    }
}

void TF1DMappingCanvas::mouseDoubleClickEvent(QMouseEvent *event) 
{
    event->accept();
    if (event->button() == Qt::LeftButton)
        changeCurrentColor();
}

void TF1DMappingCanvas::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Shift && underMouse() &&
        hitLine(QVector2D(mousePos.x(), mousePos.y())) >= 0 && !dragging)
    {
        setCursor(Qt::SizeVerCursor);
    }
}

void TF1DMappingCanvas::keyReleaseEvent(QKeyEvent* event)
{
    unsetCursor();
    if (event->key() == Qt::Key_Delete && selectedKey != 0) {
        event->accept();
        deleteKey();
    }
}

//--------- slots ---------//

void TF1DMappingCanvas::changeCurrentColor(const QColor& cc) 
{
    if (!selectedKey || !cc.isValid())
        return;

	QColor c = cc;
    bool changedColor = false;
    if (selectedKey->isSplit() && !selectedLeftPart) {
		c.setAlpha(selectedKey->getColorR().alpha());
        if (selectedKey->getColorR() != c) {
            selectedKey->setColorR(c);
            changedColor = true;
        }
    }
    else {
		c.setAlpha(selectedKey->getColorL().alpha());
        if (selectedKey->getColorL() != c) {
            selectedKey->setColorL(c);
            changedColor = true;
        }
    }

    if (changedColor) {
        update();
        emit changed();
    }
}

void TF1DMappingCanvas::splitMergeKeys()
{
    if (!selectedKey)
        return;

    selectedKey->setSplit(!selectedKey->isSplit());
    update();
    emit changed();
}

void TF1DMappingCanvas::zeroKey() 
{
    if (!selectedKey)
        return;

    TF1DMappingKey* otherKey = getOtherKey(selectedKey, selectedLeftPart);
    if (otherKey) {
        if (!otherKey->isSplit())
            otherKey->setSplit(true);
        if (selectedLeftPart)
            otherKey->setAlphaR(0.0);
        else
            otherKey->setAlphaL(0.0);
    }

    if (!selectedKey->isSplit())
        selectedKey->setSplit(true);

    if (selectedLeftPart)
        selectedKey->setAlphaL(0.0);
    else
        selectedKey->setAlphaR(0.0);

    update();
    emit changed();
}

void TF1DMappingCanvas::deleteKey() 
{
    if (!selectedKey || keys.size() < 3)
        return;

	std::vector<TF1DMappingKey *>::iterator keyIterator = std::find(keys.begin(), keys.end(), selectedKey);
    if (keyIterator != keys.end())
        keys.erase(keyIterator);
    delete selectedKey;
    selectedKey = 0;

    update();
    emit changed();
}

void TF1DMappingCanvas::resetTransferFunc()
{
    selectedKey = 0;
    emit resetTransferFunction();
    update();
}

void TF1DMappingCanvas::toggleClipThresholds()
{
    clipThresholds = !clipThresholds;
    if (clipThresholds)
        xRange = QVector2D(thresholdL, thresholdU);
    else
        xRange = QVector2D(0.f, 1.f);

    //histogramPainter->setxRange(xRange);

    update();
}

//--------- protected helper functions ---------//

void TF1DMappingCanvas::changeCurrentColor() 
{
    if (!selectedKey || noColor)
        return;

    QColor oldColor;
    if (selectedKey->isSplit() && !selectedLeftPart)
        oldColor = selectedKey->getColorR();
    else
        oldColor = selectedKey->getColorL();

    QColor newColor = QColorDialog::getColor(oldColor, 0);
    if (newColor.isValid())
        changeCurrentColor(newColor);
}

void TF1DMappingCanvas::insertNewKey(QVector2D& hit) 
{
    hit.setX(Clamp(hit.x(), 0.f, 1.f));
	hit.setY(Clamp(hit.y(), 0.f, 1.f));

    TF1DMappingKey* key = new TF1DMappingKey(hit.x(), Qt::lightGray);

    // insert key at appropriate location
	std::vector<TF1DMappingKey *>::iterator keyIterator = keys.begin();
	// Fast-forward to the correct position
	while ((keyIterator != keys.end()) && (key->getIntensity() > (*keyIterator)->getIntensity()))
		keyIterator++;
	keys.insert(keyIterator, key);

    TF1DMappingKey* leftKey = getOtherKey(key, true);
    TF1DMappingKey* rightKey = getOtherKey(key, false);

    // interpolate color of inserted key from neighbouring keys
    // (weighted by distance)
    // the alpha value is determined by hit.y()
    QColor keyColor;
    if (!leftKey)
        keyColor = rightKey->getColorL();
    else if (!rightKey)
        keyColor = leftKey->getColorR();
    else {
        float leftSource = leftKey->getIntensity();
        float rightSource = rightKey->getIntensity();
        float distSource = rightSource - leftSource;
        QColor leftColor = leftKey->getColorR();
        QColor rightColor = rightKey->getColorL();
		float t = (distSource - (hit.x() - leftSource)) / distSource;
		keyColor.setRed(leftColor.red() * t + rightColor.red() * (1 - t));
		keyColor.setGreen(leftColor.green() * t + rightColor.green() * (1 - t));
		keyColor.setBlue(leftColor.blue() * t + rightColor.blue() * (1 - t));
		keyColor.setAlpha(leftColor.alpha() * t + rightColor.alpha() * (1 - t));
    }
    key->setColorL(keyColor);
    //overwrite alpha value with clicked position
    key->setAlphaL(hit.y());

    selectedKey = key;
}

TF1DMappingKey* TF1DMappingCanvas::getOtherKey(TF1DMappingKey* selectedKey, bool selectedLeftPart)
{
    TF1DMappingKey* otherKey = 0;
    for (int i=0; i < keys.size(); ++i) {
        if ((selectedLeftPart && i < keys.size() - 1 && keys[i + 1] == selectedKey) ||
            (!selectedLeftPart && i > 0 && keys[i - 1] == selectedKey)) {
            otherKey = keys[i];
        }
    }
    return otherKey;
}

int TF1DMappingCanvas::hitLine(const QVector2D& p) 
{
    int hit = -1;
    QVector2D sHit = QVector2D(p.x(), static_cast<float>(height()) - p.y());
    QVector2D old;
    for (int i=0; i < keys.size(); ++i) {
        TF1DMappingKey* key = keys[i];
        QVector2D p = wtos(QVector2D(key->getIntensity(), key->getColorL().alpha() / 255.f));
        if (i > 0) {
            QVector2D p1 = QVector2D(old.x() + 1.f, old.y());
            QVector2D p2 = QVector2D(p.x() - 1.f, p.y());
            float s = (p2.y() - p1.y()) / (p2.x() - p1.x());
            int a = static_cast<int>(p1.y() + (sHit.x() - p1.x()) * s);
            if ((sHit.x() >= p1.x()+10) && (sHit.x() <= p2.x()-10) && (abs(static_cast<int>(sHit.y()) - a) < 5)) {
                hit = i - 1;
            }
        }

        old = p;
        if (key->isSplit())
            old = wtos(QVector2D(key->getIntensity(), key->getColorR().alpha() / 255.f));
    }
    return hit;
}

void TF1DMappingCanvas::paintKeys(QPainter& paint)
{
    for (int i=0; i<keys.size(); ++i) {
        TF1DMappingKey *key = keys[i];
        QVector2D p = wtos(QVector2D(key->getIntensity(), key->getColorL().alpha() / 255.0));
        int props;
		QColor color = key->getColorL();
		color.setAlpha(255);
        if (key->isSplit()) {
            props = MARKER_LEFT;
            if (key == selectedKey && selectedLeftPart)
                props |= MARKER_SELECTED;

            drawMarker(paint, color, p, props);

            p = wtos(QVector2D(key->getIntensity(), key->getColorR().alpha() / 255.0));
            props = MARKER_RIGHT;
            if (key == selectedKey && !selectedLeftPart)
                props |= MARKER_SELECTED;

			color = key->getColorR();
			color.setAlpha(255);
            drawMarker(paint, key->getColorR(), p, props);
        }
        else {
            props = MARKER_NORMAL;
            if (key == selectedKey)
                props |= MARKER_SELECTED;
            drawMarker(paint, color, p, props);
        }
    }
}

void TF1DMappingCanvas::drawMarker(QPainter& paint, const QColor& color, const QVector2D& p, int props)
{
    if (noColor)
        paint.setBrush(Qt::transparent);
	else
        paint.setBrush(color);

    QPen pen(QBrush(Qt::darkGray), Qt::SolidLine);
    if (props & MARKER_SELECTED)
        pen.setWidth(3);
    paint.setPen(pen);

    if (props & MARKER_LEFT) {
        paint.drawPie(QRectF(p.x() - splitFactor * pointSize/2, p.y() - pointSize/2,
                             splitFactor * pointSize, pointSize),
                      90 * 16, 180 * 16);
    }
    else if (props & MARKER_RIGHT) {
        paint.drawPie(QRectF(p.x() - splitFactor * pointSize/2, p.y() - pointSize/2,
                             splitFactor * pointSize, pointSize),
                      270 * 16, 180 * 16);
    }
    else {
        paint.drawEllipse(QRectF(p.x() - pointSize/2, p.y() - pointSize/2,
                                 pointSize, pointSize));
    }
}


void TF1DMappingCanvas::drawHistogram(QPainter& paint)
{
    //qWarning("TF1DMappingCanvas::drawHistogram: This function is disabled now.");
	//return;
	if (m_volume == nullptr)
		return;
	if(cache == 0 || cache->rect() != rect()) //{
	{
        delete cache;
        cache = new QPixmap(rect().size());
        cache->fill(Qt::transparent);

        QPainter paint(cache);

        // put origin in lower lefthand corner
        QTransform m;
        m.translate(0.0, static_cast<float>(height())-1);
        m.scale(1.f, -1.f);
        paint.setTransform(m);
        paint.setWorldMatrixEnabled(true);
		
		// draw histogram
        paint.setPen(Qt::NoPen);
        paint.setBrush(QColor(200, 0, 0, 120));
        paint.setRenderHint(QPainter::Antialiasing, true);
		
        //TODO::
		const auto histogramWidth = 256;

        double logMaxValue = std::log(m_volume->maxIsoValue());
		double * histogram = m_volume->isoStat();
        //

		QVector2D p;
        QPointF* points = new QPointF[histogramWidth + 2];
        int count = 0;

        for (int x=0; x < histogramWidth; ++x) {
			float xpos = static_cast<float>(x) / histogramWidth;
            // Do some simple clipping here, as the automatic clipping of drawPolygon()
            // gets very slow if lots of polygons have to be clipped away, e.g. when
            // zooming to small part of the histogram.
            if (xpos >= xRange[0] && xpos <= xRange[1]) {
				float value = histogram[x] > 0 ? log(histogram[x]) / logMaxValue : 0;
				if(value > 1) value = 1;
                p = wtos(QVector2D(xpos, value * (yRange[1] - yRange[0]) + yRange[0]));

                // optimization: if the y-coord has not changed from the two last points
                // then just update the last point's x-coord to the current one
                if( (count >= 2 ) && (points[count - 2].ry() == p.y()) && (points[count - 1].ry() == p.y()) && (count >= 2) ){
                    points[count - 1].rx() = p.x();
                } else {
                    points[count].rx() = p.x();
                    points[count].ry() = p.y();
                    count++;
                }
            }
		}

        // Qt can't handle polygons that have more than 65536 points
        // so we have to split the polygon
        bool needSplit = false;
        if (count > 65536 - 2) { // 16 bit dataset
			needSplit = true;
            count = 65536 - 2; // 2 points needed for closing the polygon
        }

        if (count > 0) {
            // move x coordinate of first and last points to prevent vertical holes caused
            // by clipping
            points[0].rx() = wtos(QVector2D(xRange[0], 0.f)).x();
            if (count < histogramWidth - 2) // only when last point was actually clipped
				points[count - 1].rx() = wtos(QVector2D(xRange[1], 0.f)).x();

            // needed for a closed polygon
            p = wtos(QVector2D(0.f, yRange[0]));
            points[count].rx() = points[count - 1].rx();
            points[count].ry() = p.y();
            count++;
            p = wtos(QVector2D(0.f, yRange[0]));
            points[count].rx() = points[0].rx();
            points[count].ry() = p.y();
            count++;

            paint.drawPolygon(points, count);
		}

        // draw last points when splitting is needed
        if (needSplit && false) {
            delete[] points;
            points = new QPointF[5];
            count = 0;
            for (int x=histogramWidth - 2; x < histogramWidth; ++x) {
                float xpos = static_cast<float>(x) / histogramWidth;
                if (xpos >= xRange[0] && xpos <= xRange[1]) {
                    float value = log(histogram[x]) / logMaxValue;
					if(value > 1) value = 1;
                    p = wtos(QVector2D(xpos, value * (yRange[1] - yRange[0]) + yRange[0]));
                    points[x-histogramWidth+3].rx() = p.x();
                    points[x-histogramWidth+3].ry() = p.y();
                    count++;
                }
             }
             if (count > 0) {
				// move x coordinate of last point to prevent vertical holes caused by clipping
                points[count - 1].rx() = wtos(QVector2D(xRange[1], 0.f)).x();

                // needed for a closed polygon
				p = wtos(QVector2D(0.f, yRange[0]));
                points[count].rx() = points[count - 1].rx();
                points[count].ry() = p.y();
                count++;
                p = wtos(QVector2D(0, yRange[0]));
                points[count].rx() = points[0].rx();
                points[count].ry() = p.y();
                count++;

                paint.drawPolygon(points, 5);
            }
        }
        delete[] points;
    }

    paint.drawPixmap(0, 0, *cache);
}


QVector2D TF1DMappingCanvas::wtos(QVector2D p) 
{
    float sx = (p.x() - xRange[0]) / (xRange[1] - xRange[0]) * (static_cast<float>(width())  - 2 * padding - 1.5 * arrowLength) + padding;
    float sy = (p.y() - yRange[0]) / (yRange[1] - yRange[0]) * (static_cast<float>(height()) - 2 * padding - 1.5 * arrowLength) + padding;
    return QVector2D(sx, sy);
}

QVector2D TF1DMappingCanvas::stow(QVector2D p) 
{
    float wx = (p.x() - padding) / (static_cast<float>(width())  - 2 * padding - 1.5 * arrowLength) * (xRange[1] - xRange[0]) + xRange[0];
    float wy = (p.y() - padding) / (static_cast<float>(height()) - 2 * padding - 1.5 * arrowLength) * (yRange[1] - yRange[0]) + yRange[0];
    return QVector2D(wx, wy);
}

//--------- additional functions ---------//

QSize TF1DMappingCanvas::minimumSizeHint () const 
{
    return QSize(280,140);
}

QSize TF1DMappingCanvas::sizeHint () const 
{
    return QSize(280, 140);
}

QSizePolicy TF1DMappingCanvas::sizePolicy () const 
{
    return QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
}

void TF1DMappingCanvas::createStdFunc()
{
	for(size_t i = 0; i < keys.size(); ++i)
		delete keys[i];
	keys.clear();

	keys.push_back(new TF1DMappingKey(0.f, QColor(0, 0, 0, 0)));
    keys.push_back(new TF1DMappingKey(1.f, QColor(255, 255, 255, 255)));
}


void TF1DMappingCanvas::setThreshold(float l, float u)
{
    thresholdL = l;
    thresholdU = u;
    if (clipThresholds)
        xRange = QVector2D(thresholdL, thresholdU);
    update();
}

QVector2D TF1DMappingCanvas::getThresholds() const
{
	return QVector2D(thresholdL, thresholdU);
}

void TF1DMappingCanvas::hideCoordinates()
{
    QToolTip::hideText();
}

void TF1DMappingCanvas::updateCoordinates(QPoint pos, QVector2D values)
{
    std::ostringstream os;
    os.precision(2);
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << values.x()*255 << " / " << values.y()*255.f;
    QToolTip::showText(mapToGlobal(pos), QString(os.str().c_str()));
}

void TF1DMappingCanvas::setXAxisText(const std::string& text) 
{
    xAxisText = QString(text.c_str());
}

void TF1DMappingCanvas::setYAxisText(const std::string& text) 
{
    yAxisText = QString(text.c_str());
}


void TF1DMappingCanvas::getTransferFunction(float* transferFunction, size_t dimension, float factor)
{
    int frontEnd = static_cast<int>(std::floor(thresholdL * dimension + 0.5));
    int backStart = static_cast<int>(std::floor(thresholdU * dimension + 0.5));
    //all values before front_end and after back_start are set to zero
    //all other values remain the same
	for(int x = 0; x < frontEnd; ++x) {
		transferFunction[x * 4 + 0] = 0;
		transferFunction[x * 4 + 1] = 0;
		transferFunction[x * 4 + 2] = 0;
		transferFunction[x * 4 + 3] = 0;
	}

	float r, g, b, a;
    std::vector<TF1DMappingKey*>::const_iterator keyIterator = keys.begin();
	// iterate through all keys until we get to the correct position
	for(int x = frontEnd; x < backStart; ++x) {
		float value = static_cast<float>(x) / (dimension - 1);
		while ((keyIterator != keys.end()) && (value > (*keyIterator)->getIntensity()))
			keyIterator++;
		if (keyIterator == keys.begin()) {
			QColor color = keys[0]->getColorL();
			r = color.red() / 255.f;
			g = color.green() / 255.f;
			b = color.blue() / 255.f;
			a = color.alpha() / 255.f;
		}
		else if (keyIterator == keys.end()) {
			QColor color = (*(keyIterator-1))->getColorR();
			r = color.red() / 255.f;
			g = color.green() / 255.f;
			b = color.blue() / 255.f;
			a = color.alpha() / 255.f;
		} else{
			// calculate the value weighted by the destination to the next left and right key
			TF1DMappingKey* leftKey  = *(keyIterator-1);
			TF1DMappingKey* rightKey = *keyIterator;
			float fraction   = (value - leftKey->getIntensity()) / (rightKey->getIntensity() - leftKey->getIntensity());
			QColor leftDest  = leftKey->getColorR();
			QColor rightDest = rightKey->getColorL();
			r = leftDest.red()   / 255.0 + (rightDest.red()   - leftDest.red())   / 255.f * fraction;
			g = leftDest.green() / 255.0 + (rightDest.green() - leftDest.green()) / 255.f * fraction;
			b = leftDest.blue()  / 255.0 + (rightDest.blue()  - leftDest.blue())  / 255.f * fraction;
			a = leftDest.alpha() / 255.0 + (rightDest.alpha() - leftDest.alpha()) / 255.f * fraction;
		}
		if(factor != 1)
			a = 1.0 - pow(1.f - a, factor);
		transferFunction[x * 4 + 0] = r;
		transferFunction[x * 4 + 1] = g;
		transferFunction[x * 4 + 2] = b;
		transferFunction[x * 4 + 3] = a;
	}

	//for(int x = frontEnd; x < backStart; ++x) {
	//	QColor color = getMappingForValue(static_cast<float>(x) / 256);
	//	transferFunction[x][0] = color.red();
	//	transferFunction[x][1] = color.green();
	//	transferFunction[x][2] = color.blue();
	//	transferFunction[x][3] = color.alpha();
	//}

    for(int x = backStart; x < dimension; ++x) {
		transferFunction[x * 4 + 0] = 0;
		transferFunction[x * 4 + 1] = 0;
		transferFunction[x * 4 + 2] = 0;
		transferFunction[x * 4 + 3] = 0;
	}

	//FILE* fp = fopen("TF1D4old.txt", "w");
	//for (int x = 0; x < dimension; ++x) {
	//	int r = int(transferFunction[x * 4 + 0] * 255 + 0.5);
	//	int g = int(transferFunction[x * 4 + 1] * 255 + 0.5);
	//	int b = int(transferFunction[x * 4 + 2] * 255 + 0.5);
	//	int a = int(transferFunction[x * 4 + 3] * 255 + 0.5);
	//	fprintf(fp, "%d %d %d %d\n", r, g, b, a);
	//}
	//fclose(fp);
}

QColor TF1DMappingCanvas::getMappingForValue(float value) const 
{
    // Restrict value to [0,1]
    value = (value < 0.f) ? 0.f : value;
    value = (value > 1.f) ? 1.f : value;

    // iterate through all keys until we get to the correct position
    std::vector<TF1DMappingKey*>::const_iterator keyIterator = keys.begin();

    while ((keyIterator != keys.end()) && (value > (*keyIterator)->getIntensity()))
        keyIterator++;

    if (keyIterator == keys.begin())
        return keys[0]->getColorL();
    else if (keyIterator == keys.end())
        return (*(keyIterator-1))->getColorR();
    else{
        // calculate the value weighted by the destination to the next left and right key
        TF1DMappingKey* leftKey  = *(keyIterator-1);
        TF1DMappingKey* rightKey = *keyIterator;
        float fraction   = (value - leftKey->getIntensity()) / (rightKey->getIntensity() - leftKey->getIntensity());
        QColor leftDest  = leftKey->getColorR();
        QColor rightDest = rightKey->getColorL();
        QColor result    = leftDest;
        result.setRed(result.red()     + static_cast<int>((rightDest.red()   - leftDest.red())   * fraction));
        result.setGreen(result.green() + static_cast<int>((rightDest.green() - leftDest.green()) * fraction));
        result.setBlue(result.blue()   + static_cast<int>((rightDest.blue()  - leftDest.blue())  * fraction));
        result.setAlpha(result.alpha() + static_cast<int>((rightDest.alpha() - leftDest.alpha()) * fraction));
        return result;
    }
}

void TF1DMappingCanvas::save(const char* filename)
{
	FILE *fp = fopen(filename, "w");
	if(!fp) {
		QMessageBox::critical(this, tr("Error"),
                              tr("The transfer function could not be saved."));
		return;
	}

	fprintf(fp, "%d %f %f\n", keys.size(), thresholdL, thresholdU);
	for(size_t i = 0; i < keys.size(); ++i) {
		QColor colorL = keys[i]->getColorL();
		QColor colorR = keys[i]->getColorR();
		fprintf(fp, "%f %d %d %d %d %d %d %d %d\n", keys[i]->getIntensity(),
			colorL.red(), colorL.green(), colorL.blue(), colorL.alpha(),
			colorR.red(), colorR.green(), colorR.blue(), colorR.alpha());
	}

	fclose(fp);
}

void TF1DMappingCanvas::load(const char* filename)
{
	FILE *fp = fopen(filename, "r");
	if(!fp) {
		QMessageBox::critical(this, tr("Error"),
							  "The selected transfer function could not be loaded.");
		return;
	}

	for(size_t i = 0; i < keys.size(); ++i)
		delete keys[i];
	keys.clear();
	
	int keyNum;
	fscanf(fp, "%d %f %f\n", &keyNum, &thresholdL, &thresholdU);

	float intensity;
	int rl, gl, bl, al, rr, gr, br, ar;
	for(int i = 0; i < keyNum; ++i) {
		fscanf(fp, "%f %d %d %d %d %d %d %d %d\n", &intensity, 
			&rl, &gl, &bl, &al, &rr, &gr, &br, &ar);
		TF1DMappingKey* key = new TF1DMappingKey(intensity, QColor(rl, gl, bl, al));
		if(rl != rr || gl != gr || bl != br || al != ar) {
			key->setSplit(true);
			key->setColorR(QColor(rr, gr, br, ar));
		}
		keys.push_back(key);
	}

	fclose(fp);
}
