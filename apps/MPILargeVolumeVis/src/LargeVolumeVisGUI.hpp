//
// Created by wyz on 2021/7/22.
//

#pragma once
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/volume.hpp>
#include <VolumeSlicer/volume_cache.hpp>
using namespace vs;

#include <cstdint>
#include <memory>

struct SDL_Window;
struct SDL_Renderer;
using SDL_GLContext=void *;

class WindowManager;
class Shader;

class LargeVolumeVisGUI{
public:
    LargeVolumeVisGUI();
    ~LargeVolumeVisGUI();

    void init(const char*);

    void show();

private:
    void initSDL();

    void render_imgui();

    void initRendererResource();

    void createScreenQuad();

    void createGLTexture();
private:
    SDL_Window* sdl_window = nullptr;
    SDL_GLContext gl_context = nullptr;
    uint32_t window_w = 0,window_h = 0;
    std::unique_ptr<WindowManager> window_manager;

    std::shared_ptr<CompVolume> comp_volume;
//    float volume_space_x,volume_space_y,volume_space_z;
//    float base_space;
    std::unique_ptr<ICompVolumeRenderer> comp_volume_renderer;

    uint32_t screen_quad_vao,screen_quad_vbo;
    uint32_t comp_render_tex;
    std::unique_ptr<Shader> comp_render_shader;
};


