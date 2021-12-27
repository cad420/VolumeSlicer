//
// Created by wyz on 2021/6/8.
//

#pragma once

#define VS_START namespace vs{
#define VS_END }

#ifdef _WIN32
#define VS_EXPORT __declspec(dllexport)
#else
#define VS_EXPORT
#endif

#define C_DECLARE_FUNC extern "C"


