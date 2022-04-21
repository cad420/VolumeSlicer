//
// Created by wyz on 2021/8/27.
//
#pragma once

#include <spdlog/spdlog.h>

#define LOG_DEBUG(str, ...) spdlog::debug(str, ##__VA_ARGS__)
#define LOG_INFO(str, ...) spdlog::info(str, ##__VA_ARGS__)
#define LOG_ERROR(str, ...) spdlog::error(str, ##__VA_ARGS__)
#define LOG_CRITICAL(str, ...) spdlog::critical(str, ##__VA_ARGS__)

#define SET_LOG_LEVEL_DEBUG spdlog::set_level(spdlog::level::debug);

#define SET_LOG_LEVEL_INFO spdlog::set_level(spdlog::level::info);

#define SET_LOG_LEVEL_ERROR spdlog::set_level(spdlog::level::err);

#define SET_LOG_LEVEL_CRITICAL spdlog::set_level(spdlog::level::critical);

