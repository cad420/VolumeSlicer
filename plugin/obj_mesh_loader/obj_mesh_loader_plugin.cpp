//
// Created by wyz on 2021/11/5.
//
#include "obj_mesh_loader_plugin.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <Utils/logger.hpp>
#include <spdlog/stopwatch.h>



static std::vector<std::string> GetAllPath(const std::string& config_file);
VS_START
ObjMeshLoaderPlugin::~ObjMeshLoaderPlugin()
{

}
static std::string GetFileExt(const std::string& path){
    auto pos = path.find_last_of('.');
    return path.substr(pos);
}
void ObjMeshLoaderPlugin::Open(std::string path)
{

    auto ext= GetFileExt(path);
    if(ext==".obj"){
        ReadObj(path);
    }
    else if(ext==".json"){
        auto paths = GetAllPath(path);
        if(paths.empty()){
            LOG_ERROR("Mesh config file has 0 path for obj");
        }
        for(auto& _path:paths){
            ReadObj(_path);
        }
    }
    else{
        throw std::runtime_error("ERROR: Mesh file unsupported format!");
    }

    LOG_INFO("mesh loader surface num: {0}",surfaces.size());
}
auto ObjMeshLoaderPlugin::GetSurfaces() -> const std::vector<Surface> &
{
    return surfaces;
}
void ObjMeshLoaderPlugin::Close()
{
    this->surfaces.clear();
}

void ObjMeshLoaderPlugin::ReadObj(const std::string &path)
{
    spdlog::stopwatch sw;
    tinyobj::ObjReader obj_reader;
    if(!obj_reader.ParseFromFile(path)){
        if(!obj_reader.Error().empty()){
            spdlog::error("TinyObjReader error: {0}",obj_reader.Error());
        }
        spdlog::error("Read obj file failed");
        exit(-1);
    }
    if(!obj_reader.Warning().empty()){
        spdlog::warn("TinyObjReader warning: {0}",obj_reader.Warning());
    }
    spdlog::info("Stopwatch load phrase 1: {} seconds",sw);
    sw.reset();

    auto& attrib = obj_reader.GetAttrib();
    auto& shapes = obj_reader.GetShapes();

    for(auto& shape:shapes){
        Surface surface;
        surface.name=shape.name;
        for(auto& idx:shape.mesh.indices){
            surface.indices.emplace_back(idx.vertex_index);
            if(!attrib.normals.empty()){
                surface.has_normal=true;
            }
            else{
                surface.has_normal=false;
            }
            if(!attrib.texcoords.empty()){
                surface.has_texcoord=true;
            }
            else{
                surface.has_texcoord= false;
            }
            if(surface.has_normal){
                assert(attrib.vertices.size()==attrib.normals.size());
            }
            if(surface.has_texcoord){
                assert(attrib.vertices.size()==attrib.texcoords.size());
            }
        }
        for(size_t i=0;i<attrib.vertices.size()/3;i++){
            Vertex vertex;
            vertex.pos= {attrib.vertices[i*3+0],attrib.vertices[i*3+1],attrib.vertices[i*3+2]};
            if(surface.has_normal){
                vertex.normal={attrib.normals[i*3+0],attrib.normals[i*3+1],attrib.normals[i*3+2]};
            }
            if(surface.has_texcoord){
                vertex.texcoord={attrib.texcoords[i*3+0],attrib.texcoords[i*3+1],attrib.texcoords[i*3+2]};
            }
            surface.vertices.emplace_back(vertex);
        }
        this->surfaces.emplace_back(std::move(surface));
    }
    spdlog::info("Stopwatch load phrase 2: {} seconds",sw);
}

VS_END
#include <json.hpp>
std::vector<std::string> GetAllPath(const std::string& config_file)
{
    std::ifstream in(config_file);
    if(!in.is_open()){
        throw std::runtime_error("mesh config file open failed");
    }
    nlohmann::json j;
    in>>j;
    if(j.find("neurons")==j.end()){
        throw std::runtime_error("wrong config file format");
    }
    auto neurons = j.at("neurons");
    if(neurons.find("count")==neurons.end() || neurons.find("path")==neurons.end()){
        throw std::runtime_error("invalid config file format");
    }
    int count = neurons.at("count");
    std::vector<std::string> paths = neurons.at("path");
    if(count!=paths.size()){
        LOG_ERROR("count{0} not equal to path num {1}",count,paths.size());
    }
    return std::vector<std::string>(paths.begin(),paths.begin()+ (count>paths.size()?paths.size():(std::max)(0,count)));
}


VS_REGISTER_PLUGIN_FACTORY_IMPL(ObjMeshLoaderPluginFactory)
EXPORT_PLUGIN_FACTORY_IMPL(ObjMeshLoaderPluginFactory)