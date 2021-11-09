//
// Created by csh on 10/21/2021.
//

#ifndef QTOFFSCREENRENDEREDITOR_MULTISLIDER_H
#define QTOFFSCREENRENDEREDITOR_MULTISLIDER_H

#include <QtWidgets>
#include <spdlog/spdlog.h>

/**
 * MultiSlider is a QWidget providing multiple sliders on one ridge.
 * The sliders can be moved by mouse interaction or by \a setValue()
 * method. The range of the values is from 0.0 to 1.0 and stored in float variables.
 * To avoid an overlay of the sliders the active one shunts the inactive sliders.
 */

class MultiSlider: public QWidget{
    Q_OBJECT
  public:
    MultiSlider(QWidget* parent = nullptr);

    void init(float in_stepLength = 0.01f);

    int getActivatedSliderIndex() const;

    /**
     * add a new slider at the end
     */
    void addSlider();

    /**
     * delete the activated slider
     */
    void deleteSlider();

    /**
      * raise the activated slider to the top
      *
      * * @param index the index of the slider
      */
    void sliderActivated(int index);

    /**
      * update the value of a slider
      *
      * * @param index the index of the slider
      * * @param value the new  value of the slider
      */
    void setValue(int index, float value);

    void setStepLength(float in_stepLength);

    void reset();

  protected:
    void paintEvent(QPaintEvent* event);

    void mousePressEvent(QMouseEvent* event);

    void mouseMoveEvent(QMouseEvent* event);

    void mouseReleaseEvent(QMouseEvent* event);



  public slots:


  signals:
    /**
      * This signal is emitted when the user drags a slider.
      *
      * @param index the index of the slider
      * @param value the new value of the slider
      */
    void changeValueSignal(int index, float value);

    /**
     * This signal is emitted when the user presses a slider.
     *
     * @param index the index of the slider
     */
    void activateSliderSignal(int index);

    void toggleInteraction();

  private:

    std::map<int, float> sliderValues;
    float maxValue; //value of the line's end point, set to 1.0
    float minValue; //value of the line's start point, set  to 0
    float stepLength;  //equal to the step length of spinbox and the minimal difference between 2 sliders
    int activatedSliderIndex;

    int rectHeight;
    int leftOffset;
    int rightOffset;
    int sliderWidth;

    bool ifMouseChangeValue;
};

#endif // QTOFFSCREENRENDEREDITOR_MULTISLIDER_H
