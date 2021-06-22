//
// Created by wyz on 2021/6/8.
//

#ifndef VOLUMESLICER_EXPORT_HPP
#define VOLUMESLICER_EXPORT_HPP

#define VS_START namespace vs{
#define VS_END }

#define VS_EXPORT __declspec(dllexport)

#define C_DECLARE_FUNC extern "C"

#endif //VOLUMESLICER_EXPORT_HPP
