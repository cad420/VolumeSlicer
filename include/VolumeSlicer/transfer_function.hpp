//
// Created by wyz on 2021/6/15.
//

#ifndef VOLUMESLICER_TRANSFER_FUNCTION_HPP
#define VOLUMESLICER_TRANSFER_FUNCTION_HPP
#include<VolumeSlicer/export.hpp>
#include<VolumeSlicer/status.hpp>
#include<VolumeSlicer/define.hpp>
#include<array>
#include<vector>
VS_START

class TFPoint{
public:
    TFPoint(uint8_t key,std::array<double,4> value):key(key),value(value){}
    uint8_t key;//0-255
    std::array<double,4> value;//normalize 0.0-1.0
};

class TransferFunc{
public:
    TransferFunc()=default;
    TransferFunc(TransferFunc&& tf){
        *this = std::move(tf);
    }
    TransferFunc& operator=(TransferFunc&& tf){
        this->points=std::move(tf.points);
        return *this;
    }
    std::vector<TFPoint> points;
};

VS_END

#endif //VOLUMESLICER_TRANSFER_FUNCTION_HPP
