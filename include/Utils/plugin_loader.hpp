//
// Created by wyz on 2021/8/27.
//
#pragma once
#include <Ext/plugin.hpp>
#include <unordered_map>
#include <functional>
#include <Utils/logger.hpp>
VS_START

//register plugin in self-src
class VS_EXPORT Register_PluginFactory{
  public:
    Register_PluginFactory(std::function<IPluginFactory* ()> func);
};

class VS_EXPORT PluginLoader final{
  public:
    template <typename T>
    static T* CreatePlugin(const std::string& key){
        auto& f = PluginLoader::GetPluginLoader()->factories;
        auto it = f.find(module_id_traits<T>::GetModuleID());
        if(it == f.end()){
            return nullptr;
        }
        for(auto& f:it->second){
            if(key == f()->Key()){
                LOG_INFO("Create plugin: {0}",typeid(T).name());
                return reinterpret_cast<T*>(f()->Create(key));
            }
        }
        return nullptr;
    }
    static PluginLoader* GetPluginLoader();
    static void LoadPlugins(const std::string& directory);
  private:
    PluginLoader() = default;
    std::unordered_map<std::string,std::vector<std::function<IPluginFactory*()>>> factories;
    friend class Register_PluginFactory;
};


VS_END
