//
// Created by wyz on 2021/10/6.
//
#include <Utils/plugin_loader.hpp>
#include <Utils/library_reposity.hpp>
VS_START
Register_PluginFactory::Register_PluginFactory(std::function<IPluginFactory *()> func)
{
    if( func != nullptr){
        auto id = func()->GetModuleID();
        PluginLoader::GetPluginLoader()->factories[id].push_back(func);
    }
}

PluginLoader* PluginLoader::GetPluginLoader()
{
    static PluginLoader plugin_loader;
    return &plugin_loader;
}
void PluginLoader::LoadPlugins(const std::string &directory)
{
    LibraryReposity::GetLibraryRepo()->AddLibraries(directory);

    auto& repo=LibraryReposity::GetLibraryRepo()->GetLibrepo();
    for(auto& lib : repo){
        void* sym = nullptr;
        if( (sym = lib.second->Symbol("GetPluginFactoryInstance")) != nullptr){
            LOG_INFO("Find symbol GetPluginFactoryInstance");
            auto func = reinterpret_cast<GetPluginFactory>(sym);
            auto id   = func()->GetModuleID();
            GetPluginLoader()->factories[id].push_back(func);
        }
    }
}
VS_END

