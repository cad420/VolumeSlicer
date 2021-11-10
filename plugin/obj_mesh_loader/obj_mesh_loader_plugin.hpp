//
// Created by wyz on 2021/11/5.
//
#pragma once
#include <Ext/imesh_loader_plugin_interface.hpp>
#include <Ext/plugin.hpp>
VS_START
class ObjMeshLoaderPlugin: public IMeshLoaderPluginInterface{
  public:
    using Vertex = IMeshLoaderPluginInterface::Vertex;
    using Surface = IMeshLoaderPluginInterface::Surface;
    ObjMeshLoaderPlugin() = default;

    ~ObjMeshLoaderPlugin();

    void Open(std::string path) override;

    auto GetSurfaces() -> const std::vector<Surface>& override;

    void Close() override;

  private:
    void ReadObj(const std::string& name,const std::string& path,const std::array<float,4>& color);
  private:
    std::vector<Surface> surfaces;
};

VS_END

class ObjMeshLoaderPluginFactory: public vs::IPluginFactory{
  public:
    std::string Key() const override { return ".obj"; }
    void* Create(std::string const& key) override{
        if(key == ".obj"){
            return new vs::ObjMeshLoaderPlugin();
        }
        return nullptr;
    }
    std::string GetModuleID() const override{
        return "VolumeSlicer.mesh.loader";
    }
};

VS_REGISTER_PLUGIN_FACTORY_DECL(ObjMeshLoaderPluginFactory)
EXPORT_PLUGIN_FACTORY_DECL(ObjMeshLoaderPluginFactory)