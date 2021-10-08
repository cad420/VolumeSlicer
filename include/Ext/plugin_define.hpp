//
// Created by wyz on 2021/9/29.
//
#pragma once

#include <VolumeSlicer/export.hpp>

#define EXPORT_PLUGIN_FACTORY_DECL(plugin_factory_typename) \
extern "C" VS_EXPORT ::vs::IPluginFactory* GetPluginFactoryInstance();

#define EXPORT_PLUGIN_FACTORY_IMPL(plugin_factory_typename) \
::vs::IPluginFactory* GetPluginFactoryInstance() \
{                                                           \
      return GetHelper_##plugin_factory_typename()  ;        \
}

#define VS_REGISTER_PLUGIN_FACTORY_DECL(plugin_factory_typename) \
extern "C" ::vs::IPluginFactory* GetHelper_##plugin_factory_typename();

#define VS_REGISTER_PLUGIN_FACTORY_IMPL(plugin_factory_typename) \
::vs::IPluginFactory* GetHelper_##plugin_factory_typename(){      \
    static plugin_factory_typename factory;                       \
    return &factory;                                              \
}

#define DECLARE_PLUGIN_MODULE_ID(plugin_interface_typename,module_id) \
template <> \
struct module_id_traits<plugin_interface_typename> \
{                                                                     \
     static std::string GetModuleID(){return module_id;}                                                                 \
};


VS_START
template <typename T>
struct module_id_traits;
VS_END