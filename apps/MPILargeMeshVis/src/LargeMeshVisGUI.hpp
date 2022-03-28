//
// Created by wyz on 2021/11/5.
//
#pragma once
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/mesh.hpp>
using namespace vs;

struct SDL_Window;
struct SDL_Renderer;
using SDL_GLContext=void *;

class WindowManager;

class LargeMeshVisGUI{
  public:
    LargeMeshVisGUI();
    ~LargeMeshVisGUI();

    void init(const char*);
    void show();

  private:
    void initSDL();

    void initResource();
  private:
    uint32_t window_w,window_h;

    SDL_Window* sdl_window = nullptr;
    SDL_Renderer* sdl_renderer = nullptr;

    std::unique_ptr<WindowManager> window_manager;

    float space_x,space_y,space_z;
    float base_space;

    std::shared_ptr<Mesh> mesh;
    std::unique_ptr<IMeshRenderer> mesh_renderer;
};