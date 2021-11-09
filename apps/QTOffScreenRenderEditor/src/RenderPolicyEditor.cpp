//
// Created by csh on 10/21/2021.
//

#include "RenderPolicyEditor.h"


RenderPolicyEditor::RenderPolicyEditor(QWidget* parent):QWidget(parent),
      spinBoxWidth(35),
      spinBoxHeight(20),
      maxLodNum(9)
{
    maxValue = 0.f;
    stepLength = 0.1f;

    auto widgetLayout = new QHBoxLayout;

    auto groupBoxLayout = new QVBoxLayout;
    auto renderPolicy = new QGroupBox("Render Policy");
    renderPolicy->setLayout(groupBoxLayout);

    multiSlider = new MultiSlider(this);
    groupBoxLayout->addWidget(multiSlider);

    spinBoxWidget = new QWidget(this);
    spinBoxWidget->setFixedSize(340, 80);
    groupBoxLayout->addWidget(spinBoxWidget);

    auto buttonLayout =  new QHBoxLayout;
    addLodButton = new QPushButton("add");
    addLodButton->setEnabled(false);
    deleteLodButton = new QPushButton("delete");
    deleteLodButton->setEnabled(false);
    buttonLayout->addWidget(addLodButton, 1);
    buttonLayout->addWidget(deleteLodButton, 1);
    groupBoxLayout->addLayout(buttonLayout);

    widgetLayout->addWidget(renderPolicy);
    this->setLayout(widgetLayout);

    connect(addLodButton, SIGNAL(clicked()), SLOT(addLod()));
    connect(deleteLodButton, SIGNAL(clicked()), SLOT(deleteLod()));
    connect(multiSlider, &MultiSlider::activateSliderSignal, this, &RenderPolicyEditor::sliderActivatedSlotFromMultiSlider);
    connect(multiSlider, &MultiSlider::changeValueSignal, this, &RenderPolicyEditor::valueChangedSlotFromMultiSlider);
}

void RenderPolicyEditor::init(float in_maxValue)
{
    maxValue = in_maxValue;
    float sliderStepLength = 1.f / (maxValue / stepLength);
    sliderStepLength = std::round(sliderStepLength * 1000) / 1000;

    //spdlog::info("slider step length {}", sliderStepLength);
    multiSlider->init(sliderStepLength);
    addLodButton->setEnabled(true);
    deleteLodButton->setEnabled(true);
    //addLod();
}

void RenderPolicyEditor::addLod(){
    auto size = spinBoxes.size();

    if(size > maxLodNum) return;
    if(spinBoxes.find(size - 1) != spinBoxes.end()){
        if(std::abs(spinBoxes[size - 1]->value() - maxValue) < stepLength){
            spdlog::error("add lod failed. Last lod too large. Slider number :{}.", size);
            return;
        }
    }

    multiSlider->addSlider();

    auto index = size;

    if(spinBoxes.find(index) != spinBoxes.end()){
        spdlog::error("add lod failed .index {} exists.", index);
        return;
    }

    spinBoxes[index] = new QDoubleSpinBox(spinBoxWidget);
    spinBoxes[index]->setDecimals(static_cast<int>(std::log10(1 / stepLength)));
    spinBoxes[index]->setSingleStep(stepLength);
    spinBoxes[index]->setRange(0, maxValue);
    spinBoxes[index]->setValue(maxValue);
    spinBoxes[index]->move(spinBoxWidget->width() - spinBoxWidth, 0);
    spinBoxes[index]->resize(spinBoxWidth ,spinBoxHeight);
    spinBoxes[index]->setKeyboardTracking(false);
    spinBoxes[index]->setObjectName( QString::number(index));
    spinBoxes[index]->setVisible(true);

    connect(spinBoxes[index], &QDoubleSpinBox::valueChanged, this, &RenderPolicyEditor::valueChangedSlotFromSelf);

    //log::info("add lod success. Slider number:{}, new slider pos:{},{}", index + 1,spinBoxes[index]->x(),spinBoxes[index]->y());
    emit renderPolicyChanged();
}

void RenderPolicyEditor::deleteLod(){

    auto index = multiSlider->getActivatedSliderIndex();

    if(spinBoxes.find(index) == spinBoxes.end()){
        spdlog::error("delete lod failed.activated slider index:{}", index);
        return;
    }
    multiSlider->deleteSlider();

    delete spinBoxes[index];
    spinBoxes.erase(index);

    for(auto& item : spinBoxes){
        auto itemIndex = item.first;
        if(itemIndex > index){
            auto id = item.second->objectName().toInt() - 1;
            item.second->setObjectName( QString::number(id));
            auto node = spinBoxes.extract(itemIndex);
            node.key() = itemIndex - 1;
            spinBoxes.insert(std::move(node));
        }
    }

    spdlog::info("delete lod .Activated slider index:{}, slider left number: {}", index, spinBoxes.size());
    emit renderPolicyChanged();
}

void RenderPolicyEditor::getRenderPolicy(float* rp){
    std::vector<float> m_rp;

    for(auto& item : spinBoxes){
        m_rp.push_back(item.second->value());
    }
    std::sort(m_rp.begin(),m_rp.end());

    int i = 0;
    while(i < m_rp.size()){
        rp[i] = m_rp[i];
        i++;
    }
    rp[i] = std::numeric_limits<double>::max();
}

void RenderPolicyEditor::valueChangedSlotFromSelf(double value){
    auto spinBox = qobject_cast<QDoubleSpinBox*>(QObject::sender());
    int index = spinBox->objectName().toInt();

    auto leftSpinBox = spinBoxes.find(index - 1);
    auto rightSpinBox = spinBoxes.find(index + 1);
    if(leftSpinBox != spinBoxes.end() && value < leftSpinBox->second->value() + stepLength){
        value = leftSpinBox->second->value() + stepLength;
        spinBoxes[index]->setValue(value);
    }
    if(rightSpinBox != spinBoxes.end() && value > rightSpinBox->second->value() - stepLength){
        value = rightSpinBox->second->value() - stepLength;
        spinBoxes[index]->setValue(value);
    }

    //float value = spinBox->value();
    int xPos = static_cast<int>(value / maxValue * spinBoxWidget->width() - spinBoxWidth / 2);
    if(xPos > spinBoxWidget->width() - spinBoxWidth) xPos = spinBoxWidget->width() - spinBoxWidth;
    if(xPos < 0) xPos = 0;
    spinBoxes[index]->setGeometry(xPos,0, spinBoxWidth, spinBoxHeight);
    spinBoxes[index]->raise();

    float normalizedValue = value / maxValue;
    multiSlider->setValue(index, normalizedValue);

    emit renderPolicyChanged();
}

//void RenderPolicyEditor::sliderActivatedSlotFromSelf()
//{
//    auto spinBox = qobject_cast<QDoubleSpinBox*>(QObject::sender());
//    int index = spinBox->objectName().toInt();
//    spinBoxes[index]->raise();
//
//    multiSlider->sliderActivated(index);
//}

void RenderPolicyEditor::valueChangedSlotFromMultiSlider(int index, float normalizedValue){
    float value = maxValue * normalizedValue;
    value = std::floor((value + stepLength / 2) / stepLength) * stepLength;

    auto leftSpinBox = spinBoxes.find(index - 1);
    auto rightSpinBox = spinBoxes.find(index + 1);
    if(leftSpinBox != spinBoxes.end() && value < leftSpinBox->second->value() + stepLength){
        value = leftSpinBox->second->value() + stepLength;
        spinBoxes[index]->setValue(value);
    }
    if(rightSpinBox != spinBoxes.end() && value > rightSpinBox->second->value() - stepLength){
        value = rightSpinBox->second->value() - stepLength;
        spinBoxes[index]->setValue(value);
    }

    spinBoxes[index]->setValue(value);

    int xPos = static_cast<int>(normalizedValue * spinBoxWidget->width() - spinBoxWidth / 2);
    if(xPos > spinBoxWidget->width() - spinBoxWidth) xPos = spinBoxWidget->width() - spinBoxWidth;
    if(xPos < 0) xPos = 0;
    spinBoxes[index]->setGeometry(xPos,0, spinBoxWidth, spinBoxHeight);
    spinBoxes[index]->raise();

    emit renderPolicyChanged();
}

void RenderPolicyEditor::sliderActivatedSlotFromMultiSlider(int index)
{
    spinBoxes[index]->raise();
}

void RenderPolicyEditor::volumeClosed()
{
    spdlog::info("{0}.",__FUNCTION__ );

    maxValue = 0.f;
    stepLength = 0.f;

    addLodButton->setEnabled(false);
    deleteLodButton->setEnabled(false);

    for(auto& item:spinBoxes){
        delete item.second;
    }
    spinBoxes.clear();

    multiSlider->reset();
}

void RenderPolicyEditor::setRenderPolicy(const float* data, int num){
    for(auto& item:spinBoxes){
        delete item.second;
    }
    spinBoxes.clear();
    multiSlider->reset();

    for(int i = 0;i < num;i++){
        if(data[i] <= maxValue){
            addLod();
            if(spinBoxes.find(i)!=spinBoxes.end())
                spinBoxes[i]->setValue(data[i]);
        }
        else
            return;
    }
}