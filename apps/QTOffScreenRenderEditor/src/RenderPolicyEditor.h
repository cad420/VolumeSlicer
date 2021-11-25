//
// Created by csh on 10/21/2021.
//

#ifndef QTOFFSCREENRENDEREDITOR_RENDERPOLICYEDITOR_H
#define QTOFFSCREENRENDEREDITOR_RENDERPOLICYEDITOR_H

#include <QtWidgets>

#include "MultiSlider.h"
#include <spdlog/spdlog.h>

class MultiSlider;

class RenderPolicyEditor:public QWidget{
 Q_OBJECT
public:
   explicit RenderPolicyEditor(QWidget* parent = nullptr);

   void init(float in_maxValue);

   void getRenderPolicy(float* rp);

   void volumeClosed();

   void setRenderPolicy(const float* data,int num);

private slots:
   void addLod();

   void deleteLod();

   void valueChangedSlotFromSelf(double value);

//   void sliderActivatedSlotFromSelf();

   void valueChangedSlotFromMultiSlider(int index, float normalizedValue);

   void sliderActivatedSlotFromMultiSlider(int index);

signals:

  void renderPolicyChanged();

private:
   MultiSlider* multiSlider;

   std::map<int, QDoubleSpinBox*> spinBoxes;

   QWidget* spinBoxWidget;

   QPushButton* addLodButton;
   QPushButton* deleteLodButton;

   float maxValue;
   float stepLength;

   int maxLodNum;//set to 9

   int spinBoxWidth;
   int spinBoxHeight;
};

#endif // QTOFFSCREENRENDEREDITOR_RENDERPOLICYEDITOR_H
