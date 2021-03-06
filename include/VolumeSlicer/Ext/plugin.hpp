//
// Created by wyz on 2021/9/29.
//
#pragma once

#include <VolumeSlicer/Common/export.hpp>
#include <VolumeSlicer/Ext/plugin_define.hpp>

#include <string>

VS_START
class IPluginFactory
{
  public:
    virtual std::string Key() const = 0;
    virtual void *Create(const std::string &key) = 0;
    virtual std::string GetModuleID() const = 0;
    virtual ~IPluginFactory() = default;
};
using GetPluginFactory = IPluginFactory *(*)();
VS_END
