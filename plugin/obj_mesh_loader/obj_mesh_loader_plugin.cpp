//
// Created by wyz on 2021/11/5.
//
#include "obj_mesh_loader_plugin.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <VolumeSlicer/Utils/logger.hpp>
#include <spdlog/stopwatch.h>

// name path color
static std::vector<std::tuple<std::string, std::string, std::array<float, 4>>> GetAllPath(const std::string &config_file);

VS_START

ObjMeshLoaderPlugin::~ObjMeshLoaderPlugin()
{
}

static std::string GetFileExt(const std::string &path)
{
    auto pos = path.find_last_of('.');
    return path.substr(pos);
}

void ObjMeshLoaderPlugin::Open(std::string path)
{

    auto ext = GetFileExt(path);
    if (ext == ".obj")
    {
        ReadObj("default", path, {1.f, 1.f, 1.f, 1.f});
    }
    else if (ext == ".json")
    {
        auto res = GetAllPath(path);
        for (auto &r : res)
        {
            ReadObj(std::get<0>(r), std::get<1>(r), std::get<2>(r));
        }
    }
    else
    {
        throw std::runtime_error("ERROR: Mesh file unsupported format!");
    }

    LOG_INFO("mesh loader surface num: {0}", surfaces.size());
}

auto ObjMeshLoaderPlugin::GetSurfaces() -> const std::vector<Surface> &
{
    return surfaces;
}

void ObjMeshLoaderPlugin::Close()
{
    this->surfaces.clear();
}

void ObjMeshLoaderPlugin::ReadObj(const std::string &name, const std::string &path, const std::array<float, 4> &color)
{
    spdlog::stopwatch sw;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromFile(path))
    {
        if (!obj_reader.Error().empty())
        {
            LOG_ERROR("TinyObjReader error: {0}", obj_reader.Error());
        }
        LOG_ERROR("Read obj file failed");
        exit(-1);
    }
    if (!obj_reader.Warning().empty())
    {
        LOG_ERROR("TinyObjReader warning: {0}", obj_reader.Warning());
    }
    LOG_INFO("Stopwatch load phrase 1: {} seconds", sw);
    sw.reset();

    auto &attrib = obj_reader.GetAttrib();
    auto &shapes = obj_reader.GetShapes();

    for (auto &shape : shapes)
    {
        Surface surface;
        //        surface.name=shape.name;
        surface.name = name;
        surface.color = color;
        for (auto &idx : shape.mesh.indices)
        {
            surface.indices.emplace_back(idx.vertex_index);
            if (!attrib.normals.empty())
            {
                surface.has_normal = true;
            }
            else
            {
                surface.has_normal = false;
            }
            if (!attrib.texcoords.empty())
            {
                surface.has_texcoord = true;
            }
            else
            {
                surface.has_texcoord = false;
            }
            if (surface.has_normal)
            {
                assert(attrib.vertices.size() == attrib.normals.size());
            }
            if (surface.has_texcoord)
            {
                assert(attrib.vertices.size() == attrib.texcoords.size());
            }
        }
        for (size_t i = 0; i < attrib.vertices.size() / 3; i++)
        {
            Vertex vertex;
            vertex.pos = {attrib.vertices[i * 3 + 0], attrib.vertices[i * 3 + 1], attrib.vertices[i * 3 + 2]};
            if (surface.has_normal)
            {
                vertex.normal = {attrib.normals[i * 3 + 0], attrib.normals[i * 3 + 1], attrib.normals[i * 3 + 2]};
            }
            if (surface.has_texcoord)
            {
                vertex.texcoord = {attrib.texcoords[i * 3 + 0], attrib.texcoords[i * 3 + 1],
                                   attrib.texcoords[i * 3 + 2]};
            }
            surface.vertices.emplace_back(vertex);
        }
        LOG_INFO("surface {0} with color ({1},{2},{3},{4})", surface.name, surface.color[0], surface.color[1],
                 surface.color[2], surface.color[3]);
        this->surfaces.emplace_back(std::move(surface));
    }
    LOG_INFO("Stopwatch load phrase 2: {} seconds", sw);
}

VS_END

#include <json.hpp>
// name path color
std::vector<std::tuple<std::string, std::string, std::array<float, 4>>> GetAllPath(const std::string &config_file)
{
    std::ifstream in(config_file);
    if (!in.is_open())
    {
        throw std::runtime_error("mesh config file open failed");
    }
    nlohmann::json j;
    in >> j;
    if (j.find("neurons") == j.end())
    {
        throw std::runtime_error("wrong config file format");
    }
    auto neurons = j.at("neurons");
    if (neurons.find("count") == neurons.end())
    {
        throw std::runtime_error("invalid config file format");
    }
    int count = neurons.at("count");
    auto resource = neurons.at("resource");
    if (count != resource.size())
    {
        LOG_ERROR("count {0} not equal to resource size {1}", count, resource.size());
    }
    bool is_same_color = neurons.at("neuron_color").at("use_same_color") == "true";
    std::array<float, 4> same_color = neurons.at("neuron_color").at("color");

    std::vector<std::tuple<std::string, std::string, std::array<float, 4>>> read_res;
    int cnt = 0;
    for (auto &res : resource)
    {
        std::string name, path;
        std::array<float, 4> color_v = {1.f, 1.f, 1.f, 1.f};
        if (res.find("name") == res.end())
        {
            name = "default" + std::to_string(cnt);
        }
        else
        {
            name = res.at("name");
        }
        if (res.find("path") == res.end())
        {
            LOG_ERROR("path not find, config file error");
            continue;
        }
        path = res.at("path");
        if (res.find("color") == res.end())
        {
            if (is_same_color)
            {
                color_v = same_color;
            }
            else
            {
                LOG_INFO("Not found color, use default color white");
            }
        }
        else
        {
            if (is_same_color)
            {
                color_v = same_color;
            }
            else
            {
                std::vector<float> color = res.at("color");
                std::copy(color.begin(), color.end(), color_v.begin());
            }
        }
        read_res.emplace_back(name, path, color_v);
        cnt++;
    }
    return read_res;
}

VS_REGISTER_PLUGIN_FACTORY_IMPL(ObjMeshLoaderPluginFactory)
EXPORT_PLUGIN_FACTORY_IMPL(ObjMeshLoaderPluginFactory)