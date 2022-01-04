//
// Created by wyz on 2021/11/5.
//

#pragma once

#include <VolumeSlicer/Ext/imesh_loader_plugin_interface.hpp>
#include <VolumeSlicer/mesh.hpp>

VS_START

class MeshImpl : public Mesh
{
  public:
    using Surface = Mesh::Surface;
    using Vertex = Mesh::Vertex;

    MeshImpl(std::string path);

    ~MeshImpl() override;

    void Transform(float space_x, float space_y, float space_z) override;

    int GetSurfaceNum() override;

    auto GetSurfaceNames() -> std::vector<std::string> override;

    const Surface &GetSurfaceByName(std::string) override;

    auto GetAllSurfaces() -> const std::vector<Surface> & override;

    void SetSurfaceColorByName(const std::string &, const std::array<float, 4> &) override;

  private:
    std::vector<Surface> surfaces;
    std::unique_ptr<IMeshLoaderPluginInterface> mesh_loader;
};

VS_END
