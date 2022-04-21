//
// Created by wyz on 2021/6/15.
//

#pragma once
#include <VolumeSlicer/Common/define.hpp>
#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Common/status.hpp>
#include <array>
#include <vector>
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
    TransferFunc(const TransferFunc&) = default;
    TransferFunc& operator=(const TransferFunc&) = default;
    TransferFunc(TransferFunc&& tf) noexcept{
        *this = std::move(tf);
    }
    TransferFunc& operator=(TransferFunc&& tf) noexcept{
        this->points=std::move(tf.points);
        return *this;
    }
    std::vector<TFPoint> points;
};

VS_END


