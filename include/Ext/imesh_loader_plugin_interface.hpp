//
// Created by wyz on 2021/11/5.
//
#pragma once
#include <Ext/plugin_define.hpp>
#include <string>
#include <vector>
#include <array>
VS_START

class IMeshLoaderPluginInterface{
  public:
    struct Vertex{
        std::array<float,3> pos;
        std::array<float,3> normal;
        std::array<float,3> texcoord;
    };
    struct Surface{
        std::string name;
        bool has_normal;
        bool has_texcoord;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
    };
    virtual void Open(std::string path) = 0;

    virtual auto GetSurfaces() -> const std::vector<Surface>& = 0;

    virtual void Close() = 0;

};

DECLARE_PLUGIN_MODULE_ID(IMeshLoaderPluginInterface,"VolumeSlicer.mesh.loader");

VS_END