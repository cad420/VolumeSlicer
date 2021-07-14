//
// Created by wyz on 2021/7/12.
//

#ifndef VOLUMESLICER_VOLUMESLICEGUI_HPP
#define VOLUMESLICER_VOLUMESLICEGUI_HPP


#include <VolumeSlicer/slice.hpp>
#include <VolumeSlicer/volume.hpp>
#include <VolumeSlicer/volume_sampler.hpp>
#include <VolumeSlicer/frame.hpp>
#include <VolumeSlicer/transfer_function.hpp>
using namespace vs;

struct SDL_Window;
struct SDL_Renderer;
using SDL_GLContext=void *;

class WindowManager;
class Shader;

class VolumeSliceGUI final{
public:
    VolumeSliceGUI();
    ~VolumeSliceGUI();
    void init(const char* config_file);
    void show();

    void set_comp_volume(const char*);
    void set_raw_volume(const char*,uint32_t,uint32_t,uint32_t);
    void set_transferfunc(TransferFunc);

private:
    void initSDL();

    void initGLResource();
    void createGLTexture();
    void createGLSampler();
    void createShader();
    void createScreenQuad();

    void render_imgui();

    void render_root_frame();
    void render_node_frame();
private:
    SDL_Window* sdl_window;
    SDL_GLContext gl_context;
    uint32_t window_w,window_h;
    SDL_Renderer* sdl_renderer;

    std::shared_ptr<CompVolume> comp_volume;
    std::shared_ptr<RawVolume> raw_volume;
    std::shared_ptr<Slicer> slicer;
    std::unique_ptr<VolumeSampler> comp_volume_sampler;
    std::unique_ptr<VolumeSampler> raw_volume_sampler;
    float volume_space_x,volume_space_y,volume_space_z;

    std::unique_ptr<WindowManager> window_manager;
    std::shared_ptr<Slicer> world_slicer;
    Slice world_slice;

    uint32_t screen_quad_vao,screen_quad_vbo;

    uint32_t comp_sample_tex;
    uint32_t tex_sampler;
    uint32_t raw_sample_tex,raw_render_tex;

    Frame comp_sample_frame;
    Frame raw_sample_frame;

    std::unique_ptr<Shader> comp_render_shader;
};


#endif //VOLUMESLICER_VOLUMESLICEGUI_HPP
