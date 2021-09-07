//
// Created by wyz on 2021/7/22.
//

#ifndef VOLUMESLICER_LARGEVOLUMEVISGUI_HPP
#define VOLUMESLICER_LARGEVOLUMEVISGUI_HPP
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
private:
    SDL_Window* sdl_window;
    SDL_GLContext gl_context;
    uint32_t window_w,window_h;
    SDL_Renderer* sdl_renderer;
    std::unique_ptr<WindowManager> window_manager;

    std::shared_ptr<CompVolume> comp_volume;
    float volume_space_x,volume_space_y,volume_space_z;
    float base_space;
    std::unique_ptr<ICompVolumeRenderer> comp_volume_renderer;

};

#endif //VOLUMESLICER_LARGEVOLUMEVISGUI_HPP
