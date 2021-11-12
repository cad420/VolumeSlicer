//
// Created by wyz on 2021/11/2.
//
#pragma once

#include <VolumeSlicer/export.hpp>

#include <memory>
#include <string>
#include <vector>
#include <array>

VS_START

class VS_EXPORT Mesh{
public:
  struct Vertex{
    std::array<float,3> pos;
    std::array<float,3> normal;
  };
  struct Surface{
      std::string name;
      std::vector<Vertex> vertices;
      bool has_normal;
      std::vector<uint32_t> indices;
      std::array<float,4> color;
  };
public:
    static std::unique_ptr<Mesh> Load(std::string path);

    virtual ~Mesh(){}

    //mesh's coord transform, like model matrix transform
    virtual void Transform(float space_x,float space_y,float space_z) = 0;

    virtual int GetSurfaceNum() = 0;

    virtual auto GetSurfaceNames() -> std::vector<std::string> = 0;

    virtual auto GetSurfaceByName(std::string) -> const Surface& = 0;

    virtual auto GetAllSurfaces() -> const std::vector<Surface>& = 0;

    virtual void SetSurfaceColorByName(const std::string&,const std::array<float,4>&) = 0;
};


VS_END
