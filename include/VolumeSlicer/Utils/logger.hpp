//
// Created by wyz on 2021/8/27.
//
#pragma once

#include <spdlog/spdlog.h>

#define LOG_INFO(str, ...) spdlog::info(str, ##__VA_ARGS__)
#define LOG_ERROR(str, ...) spdlog::error(str, ##__VA_ARGS__)
#define LOG_CRITICAL(str, ...) spdlog::critical(str, ##__VA_ARGS__)

