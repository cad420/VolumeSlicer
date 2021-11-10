//
// Created by wyz on 2021/11/5.
//
#include "mesh_impl.hpp"
#include <algorithm>
#include <cmath>
#include <Utils/plugin_loader.hpp>
VS_START

std::unique_ptr<Mesh> Mesh::Load(std::string path)
{
    return std::make_unique<MeshImpl>(std::move(path));
}

MeshImpl::MeshImpl(std::string path)
{
    PluginLoader::LoadPlugins("./plugins");
    this->mesh_loader = std::unique_ptr<IMeshLoaderPluginInterface>(
            PluginLoader::CreatePlugin<IMeshLoaderPluginInterface>(".obj")
            );
    if(!mesh_loader){
        throw std::runtime_error("Plugin for mesh loader create failed");
    }
    mesh_loader->Open(path);
    auto& shapes = mesh_loader->GetSurfaces();
    for(auto& shape:shapes){
        Surface surface;
        surface.indices=shape.indices;
        surface.has_normal=shape.has_normal;
        surface.name=shape.name;
        surface.vertices.reserve(shape.vertices.size());
        surface.color = shape.color;
        for(auto& v:shape.vertices){
            surface.vertices.emplace_back(Vertex{
                v.pos,
                v.normal
            });
        }
        this->surfaces.emplace_back(std::move(surface));
    }
    LOG_INFO("Surface num: {0}",surfaces.size());
    for(auto& s:surfaces){
        LOG_INFO("surface {0} has vertex num: {1}",s.name,s.vertices.size());
    }
    mesh_loader->Close();
}

MeshImpl::~MeshImpl()
{

}

void MeshImpl::Transform(float space_x, float space_y, float space_z)
{
    for(auto& surface:surfaces){
        std::transform(surface.vertices.begin(),surface.vertices.end(),
                       surface.vertices.begin(),
                       [space_x,space_y,space_z](Vertex v){
                            v.pos[0] *= space_x;
                            v.pos[1] *= space_y;
                            v.pos[2] *= space_z;
                            v.normal[0] *=space_x;
                            v.normal[1] *=space_y;
                            v.normal[2] *=space_z;
                            float len = std::sqrt(v.normal[0]*v.normal[0]+v.normal[1]*v.normal[1]+v.normal[2]*v.normal[2]);
                            v.normal[0] /= len;
                            v.normal[1] /= len;
                            v.normal[2] /= len;
                            return v;
                       });
    }
}

int MeshImpl::GetSurfaceNum()
{
    return surfaces.size();
}

auto MeshImpl::GetSurfaceNames() -> std::vector<std::string>
{
    std::vector<std::string> names;
    for(auto& surface:surfaces){
        names.emplace_back(surface.name);
    }
    return names;
}

const MeshImpl::Surface &MeshImpl::GetSurfaceByName(std::string name)
{
    for(auto& surface:surfaces){
        if(surface.name==name){
            return surface;
        }
    }
    return {};
}

auto MeshImpl::GetAllSurfaces() -> const std::vector<Surface> &
{
    return surfaces;
}
void MeshImpl::SetSurfaceColorByName(const std::string& name,const std::array<float,4>& color)
{
    for(auto& s:surfaces){
        if(s.name==name){
            s.color=color;
            return;
        }
    }
    LOG_ERROR("Not found surface with name {0}",name);
}

VS_END

