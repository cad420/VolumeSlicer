//
// Created by csh on 10/21/2021.
//

#include "MultiSlider.h"


MultiSlider::MultiSlider(QWidget* parent):QWidget(parent){
    setFixedHeight(30);

    maxValue = 1.f;
    minValue = 0.f;
    sliderWidth = 5;
    leftOffset = rightOffset = 5.f;
    rectHeight = 10;

    stepLength= 0.05f;
    activatedSliderIndex = -1;
    ifMouseChangeValue = false;

    //connect(this, SIGNAL(changeValueSignal()), parent, SLOT(valueChangedSlotFromMultiSlider()));
    //(this, SIGNAL(activateSliderSignal()), parent, SLOT(sliderActivatedSlotFromMultiSlider()));
}

void MultiSlider::init(float in_stepLength){
    stepLength = in_stepLength;
//    maxValue = std::ceil(in_maxValue / stepLength) * stepLength;
//    minValue = std::floor(in_minValue / stepLength) * stepLength;
}

void MultiSlider::paintEvent(QPaintEvent* event){
    event->accept();

    const QColor activatedSliderColor(255, 255, 255);
    const QColor sliderColor(195, 219, 224);
    const QColor sliderDarkColor(75, 130, 89);
    const QColor rectColor(180, 180, 180);
    const QColor rectScaleColor(10, 10, 10);
    const QColor rectBorderColor= rectScaleColor;

    QPainter paint(this);
    paint.setRenderHint(QPainter::Antialiasing);

    //draw rect
    paint.setBrush(rectColor);
    paint.setPen(rectBorderColor);
    paint.drawRect(leftOffset, 0, width() - leftOffset - rightOffset, rectHeight);

    //draw scale
    paint.setPen(rectScaleColor);
    for(int i = 0; i < 9; i++){
        int xOffset = static_cast<int>(std::floor((width() - leftOffset - rightOffset) * (i + 1) / 10 + leftOffset + 0.5));
        paint.drawLine(xOffset, 0, xOffset, rectHeight);
    }

    //draw markers
    paint.setBrush(sliderColor);
    paint.setPen(sliderDarkColor);

    for(const auto &item : sliderValues){
        if(item.first != activatedSliderIndex){
            const auto marker = static_cast<int>(std::floor((item.second - minValue) / (maxValue  - minValue) * (width() - leftOffset - rightOffset) + leftOffset + 0.5));
            QPoint slider[5] = {
                QPoint(marker - sliderWidth, static_cast<int>(0.3f * (height() - rectHeight)) + rectHeight),
                QPoint(marker - sliderWidth, height()),
                QPoint(marker + sliderWidth, height()),
                QPoint(marker + sliderWidth, static_cast<int>(0.3f * (height() - rectHeight)) + rectHeight),
                QPoint(marker, rectHeight)
            };
            paint.save();
            paint.drawConvexPolygon(slider, 5);
            paint.restore();
        }
    }

    if(sliderValues.find(activatedSliderIndex) != sliderValues.end()){
        paint.setBrush(activatedSliderColor);
        const auto marker = static_cast<int>(std::floor((sliderValues[activatedSliderIndex] - minValue) / (maxValue  - minValue) * (width() - leftOffset - rightOffset) + leftOffset + 0.5));
        QPoint slider[5] = {
            QPoint(marker - sliderWidth, static_cast<int>(0.3f * (height() - rectHeight)) + rectHeight),
            QPoint(marker - sliderWidth, height()),
            QPoint(marker + sliderWidth, height()),
            QPoint(marker + sliderWidth, static_cast<int>(0.3f * (height() - rectHeight)) + rectHeight),
            QPoint(marker, rectHeight)
        };
        paint.save();
        paint.drawConvexPolygon(slider, 5);
        paint.restore();
    }

    //spdlog::info("paint event:activated slider index {}",activatedSliderIndex);
}

void MultiSlider::mousePressEvent(QMouseEvent* event){
    event->accept();

    const int xPos = (event->pos()).x();
    const int yPos = (event->pos()).y();

    if(yPos < rectHeight) {
        activatedSliderIndex = -1;
        update();
        return;
    }

    //if mouse points to the activated sliders, stop searching
    const auto activeSlider = sliderValues.find(activatedSliderIndex);
    if(activeSlider != sliderValues.end())
    {
        const int sliderXPos = std::floor((activeSlider->second - minValue) / (maxValue  - minValue) * (width() - leftOffset - rightOffset) + leftOffset + 0.5);
        if(xPos > sliderXPos - sliderWidth && xPos < sliderXPos + sliderWidth) return;
    }

    //find the nearest pointed slider
    int nearestSliderDistance = width();
    int nearestSliderIndex = -1;
    for(const auto item : sliderValues){
        const int sliderXPos = std::floor((item.second - minValue) / (maxValue  - minValue) * (width() - leftOffset - rightOffset) + leftOffset + 0.5);
        if(xPos > sliderXPos - sliderWidth && xPos < sliderXPos + sliderWidth && std::abs(xPos - sliderXPos) < nearestSliderDistance)
        {
            nearestSliderDistance = std::abs(xPos - sliderXPos);
            nearestSliderIndex = item.first;
        }
    }

    //update activeSliderIndex
    const auto nearestSlider = sliderValues.find(nearestSliderIndex);
    if(nearestSlider != sliderValues.end()){
        activatedSliderIndex = nearestSliderIndex;
        update();
        emit activateSliderSignal(activatedSliderIndex);
    }
    else{
        activatedSliderIndex = -1;
        update();
        return;
    }
}

void MultiSlider::mouseMoveEvent(QMouseEvent *event)
{
    event->accept();

    auto slider = sliderValues.find(activatedSliderIndex);
    if(slider != sliderValues.end() && event->buttons() && Qt::LeftButton){
        const int mouseXPos = (event->pos()).x();

        float leftBorder;
        float rightBorder;
        const auto leftSlider = sliderValues.find(activatedSliderIndex - 1);
        const auto rightSlider = sliderValues.find(activatedSliderIndex + 1);

        if(leftSlider != sliderValues.end()){
            leftBorder =  leftSlider->second + stepLength;
        }else{
            leftBorder = minValue;
        }

        if(rightSlider != sliderValues.end()){
            rightBorder =  rightSlider->second - stepLength;
        }else
        {
            rightBorder = maxValue;
        }

        float curValue;
        curValue = (mouseXPos * 1.f - leftOffset) / (width() - leftOffset - rightOffset) * (maxValue - minValue) + minValue;
        curValue = std::floor((curValue + stepLength / 2) / stepLength ) * stepLength;
        if(curValue >= leftBorder && curValue <= rightBorder)
            sliderValues[activatedSliderIndex] = curValue;
        else if(curValue < leftBorder)
            sliderValues[activatedSliderIndex] = leftBorder;
        else
            sliderValues[activatedSliderIndex] = rightBorder;

        ifMouseChangeValue = true;
        //spdlog::info("mouse move event. slider {} moves to {}", activatedSliderIndex, sliderValues[activatedSliderIndex]);
        update();
        emit changeValueSignal(activatedSliderIndex, sliderValues[activatedSliderIndex]);
    }
}

void MultiSlider::mouseReleaseEvent(QMouseEvent *event)
{
    event->accept();
    if(ifMouseChangeValue && event->button() == Qt::LeftButton){
        ifMouseChangeValue = false;
        emit toggleInteraction();
    }
}

void MultiSlider::sliderActivated(int index){
    activatedSliderIndex = index;
    update();
}

void MultiSlider::addSlider()
{
    int size = sliderValues.size();

    if(sliderValues.find(size - 1) != sliderValues.end())
        if(abs(sliderValues[size - 1] - maxValue) < stepLength)
            return;

    sliderValues[size] = maxValue;
    activatedSliderIndex = size;
    update();
}

void MultiSlider::deleteSlider()
{
    auto index = activatedSliderIndex;
    if(sliderValues.find(index) == sliderValues.end()) return;
    auto slider = sliderValues.extract(index);
    for(auto item : sliderValues){
        if(item.first > index)
        {
            auto tempIndex = item.first;
            auto tempValue = item.second;
            sliderValues.erase(tempIndex);
            sliderValues[tempIndex - 1] = tempValue;
        }
    }
    activatedSliderIndex = -1;
    update();
}

int MultiSlider::getActivatedSliderIndex() const
{
    if(activatedSliderIndex >= 0)
        return activatedSliderIndex;
    else return -1;
}

void MultiSlider::setValue(int index, float value)
{
    if(sliderValues.find(index) != sliderValues.end()){
        sliderValues[index] = value;
        activatedSliderIndex = index;
        //sliderValues[index] = std::floor(value / stepLength + stepLength / 2) * stepLength;;
        update();
    }
}

void MultiSlider::reset(){
    spdlog::info("{0}.",__FUNCTION__ );
    sliderValues.clear();
    stepLength= 0.05f;
    activatedSliderIndex = -1;
    ifMouseChangeValue = false;
    update();
}