//
// Created by wyz on 2021/11/5.
//
#pragma once

#include "shader_program.hpp"

#include <GLFW/glfw3.h>

#include <VolumeSlicer/render.hpp>

VS_START
class SimpleMeshRendererImpl : public SimpleMeshRenderer
{
  public:
    SimpleMeshRendererImpl(int w, int h);

    ~SimpleMeshRendererImpl() override;

    void SetMesh(std::shared_ptr<Mesh> mesh) override;

    void SetCamera(Camera camera) override;

    void SetMPIRender(MPIRenderParameter mpi) override;

    void render() override;

    auto GetImage() -> const Image<Color4b> & override;

    void resize(int w, int h) override;

    void clear() override;

  private:
    /**
     * @note must call before every OpenGL API call if in multi-thread context.
     */
    void setCurrentCtx()
    {
        glfwMakeContextCurrent(window);
    }

    void initGL();

    void setupMeshColorMap();

    void deleteGLResource();

  private:
    Image<Color4b> image;

    GLFWwindow *window = nullptr;
    int window_w, window_h;

    Camera camera;

    std::shared_ptr<Mesh> mesh;

    std::vector<std::array<float, 4>> color_map;
    uint32_t color_map_ssbo;

    std::vector<uint32_t> surfaces_vao, surfaces_vbo, surfaces_ebo;
    std::vector<size_t> surfaces_indices_num;

    std::unique_ptr<Shader> simple_mesh_render_shader;

    MPIRenderParameter mpi;
};
VS_END