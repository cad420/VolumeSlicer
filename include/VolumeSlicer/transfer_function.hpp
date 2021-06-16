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
    uint8_t key;
    std::array<double,4> value;
};

class TransferFunc{
public:
    std::vector<TFPoint> points;
};

VS_END

#endif //VOLUMESLICER_TRANSFER_FUNCTION_HPP
