//
// Created by wyz on 2021/7/22.
//
#include "LargeVolumeVisGUI.hpp"
#include "camera.hpp"
#include <SDL.h>

#include <spdlog/spdlog.h>
#include <glad/glad.h>
#include "WindowManager.hpp"

#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define DEBUG
#ifdef DEBUG
#define SDL_EXPR(exec)                                                                                                 \
    exec;                                                                                                              \
    {                                                                                                                  \
        std::string __err_str = SDL_GetError();                                                                              \
        if (__err_str.length() > 0)                                                                                          \
        {                                                                                                              \
            spdlog::error("SDL call error: {0} in {1} function: {2} line: {3}",__err_str,__FILE__,__FUNCTION__,__LINE__);\
        }                                                                                                              \
    }

#define SDL_CHECK                                                                                                      \
    {                                                                                                                  \
        std::string __err_str = SDL_GetError();                                                                              \
        if (__err_str.length() > 0)                                                                                          \
        {                                                                                                              \
            spdlog::error("SDL call error: {0} in {1} before line: {2}",__err_str,__FILE__,__LINE__);\
        }                                                                                                               \
    }

#else
#define SDL_EXPR(exec) exec
#define SCL_CHECK
#endif
#ifdef DEBUG
#define GL_EXPR(exec) \
        {             \
            GLenum gl_err; \
            exec;     \
            if((gl_err=glGetError())!=GL_NO_ERROR){ \
                 spdlog::error("OpenGL error:{0:x} caused by {1} on line {2} of file:{3}",static_cast<unsigned int>(gl_err),#exec,__LINE__,__FILE__);     \
            }\
        };

#define GL_CHECK \
         {       \
            GLenum gl_err; \
            if((gl_err=glGetError())!=GL_NO_ERROR){     \
            spdlog::error("OpenGL error: {0} caused before  on line {1} of file:{2}",static_cast<unsigned int>(gl_err),__LINE__,__FILE__);     \
            }\
         }

#else
#define GL_EXPR(exec) exec
#define GL_CHECK
#endif

void LargeVolumeVisGUI::init(const char * config_file) {
    this->window_manager=std::make_unique<WindowManager>(config_file);
    this->window_w=window_manager->GetNodeWindowWidth();
    this->window_h=window_manager->GetNodeWindowHeight();
    SetCUDACtx(0);

    initSDL();

    initRendererResource();
}

void LargeVolumeVisGUI::show() {
    bool exit=false;
    auto process_event=[&exit,this](){
        static SDL_Event event;
        ImGui_ImplSDL2_ProcessEvent(&event);
        //camera pos according to volume dim count in voxel
        static control::FPSCamera fpsCamera({10.f,10.f,25.f});

        while(SDL_PollEvent(&event)){
            switch(event.type){
                case SDL_QUIT:{
                    exit=true;
                    break;
                }
                case SDL_WINDOWEVENT:{
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_FOCUS_GAINED:;break;
                        case SDL_WINDOWEVENT_FOCUS_LOST:;break;
                    }
                    break;
                }
                case SDL_KEYDOWN:{
                    switch (event.key.keysym.sym) {
                        case SDLK_ESCAPE:{exit=true;break;}
                        case SDLK_LCTRL:{;break;}
                        case SDLK_a:{;break;}
                        case SDLK_d:{;break;}
                        case SDLK_w:{;break;}
                        case SDLK_s:{;break;}
                        case SDLK_q:{;break;}
                        case SDLK_e:{;break;}
                        case SDLK_LEFT:
                        case SDLK_DOWN:{


                            break;
                        }
                        case SDLK_RIGHT:
                        case SDLK_UP:{

                            break;
                        }
                    }
                    break;
                }
                case SDL_KEYUP:{
                    switch (event.key.keysym.sym) {
                        case SDLK_LCTRL:{

                            break;
                        }
                    }
                    break;
                }
                case SDL_MOUSEWHEEL:{

                    break;
                }
                case SDL_MOUSEBUTTONDOWN:{

                    if(event.button.button==1){

                    }
                    break;
                }
                case SDL_MOUSEBUTTONUP:{

                    if(event.button.button==1){

                    }
                    break;
                }
                case SDL_MOUSEMOTION:{

                    break;
                }
            }
        }//end of SDL_PollEvent
        auto camera_pos=fpsCamera.getCameraPos();
        auto camera_up=fpsCamera.getCameraUp();
        auto camera_look_at=fpsCamera.getCameraLookAt();
        auto camera_zoom=fpsCamera.getZoom();
        auto camera_right=fpsCamera.getCameraRight();
        Camera camera{};
        camera.pos={camera_pos.x,camera_pos.y,camera_pos.z};
        camera.up={camera_up.x,camera_up.y,camera_up.z};
        camera.look_at={camera_look_at.x,camera_look_at.y,camera_look_at.z};
        camera.right={camera_right.x,camera_right.y,camera_right.z};
        camera.zoom=camera_zoom;
        cuda_comp_volume_renderer->SetCamera(camera);
    };
    SDL_EXPR(sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED));
    SDL_Rect rect{0,0,(int)window_w,(int)window_h};
    SDL_Texture* texture=SDL_CreateTexture(sdl_renderer,SDL_PIXELFORMAT_RGBA8888,SDL_TEXTUREACCESS_STREAMING,window_w,window_h);
    SDL_CHECK


    while(!exit){

        process_event();


        cuda_comp_volume_renderer->render();


        SDL_UpdateTexture(texture, NULL, cuda_comp_volume_renderer->GetFrame().data.data(), window_w * 4);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, texture, nullptr, &rect);
        SDL_RenderPresent(sdl_renderer);


//        render_imgui();

//        SDL_GL_SwapWindow(sdl_window);
        SDL_CHECK
    }
}
void LargeVolumeVisGUI::render_imgui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(sdl_window);
    ImGui::NewFrame();
    {
        ImGui::Begin("Large Volume Renderer");
        ImGui::Text("FPS: %.1f",ImGui::GetIO().Framerate);
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void LargeVolumeVisGUI::initSDL() {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
    {
        spdlog::critical("{0} - SDL could not initialize! SDL Error: {1}.", __FUNCTION__, SDL_GetError());
        throw std::runtime_error("SDL could not initialize");
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,6);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE,8);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_EXPR(sdl_window = SDL_CreateWindow("VolumeSlicer", window_manager->GetNodeScreenOffsetX(),window_manager->GetNodeScreenOffsetY() ,
                                           window_manager->GetNodeWindowWidth(), window_manager->GetNodeWindowHeight(), SDL_WINDOW_OPENGL|SDL_WINDOW_ALLOW_HIGHDPI));

    gl_context=SDL_GL_CreateContext(sdl_window);
    SDL_GL_MakeCurrent(sdl_window,gl_context);
    SDL_GL_SetSwapInterval(1);
    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)){
        spdlog::critical("GLAD: OpenGL load failed.");
        throw std::runtime_error("GLAD: OpenGL load failed.");
    }
    glEnable(GL_DEPTH_TEST);
    SDL_CHECK

    //init imgui for sdl2 and opengl3
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();

        ImGui::StyleColorsDark();

        ImGui_ImplSDL2_InitForOpenGL(sdl_window,gl_context);
        ImGui_ImplOpenGL3_Init();
    }
}

LargeVolumeVisGUI::~LargeVolumeVisGUI() {
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
    }
    SDL_GL_DeleteContext(gl_context);
    SDL_EXPR(SDL_DestroyRenderer(sdl_renderer));
    SDL_EXPR(SDL_DestroyWindow(sdl_window));
    SDL_EXPR(SDL_Quit());
}

LargeVolumeVisGUI::LargeVolumeVisGUI() {

}

void LargeVolumeVisGUI::initRendererResource() {
    this->comp_volume=CompVolume::Load(window_manager->GetNodeResourcePath().c_str());
    this->comp_volume->SetSpaceX(0.00032f);
    this->comp_volume->SetSpaceY(0.00032f);
    this->comp_volume->SetSpaceZ(0.001f);

    this->cuda_comp_volume_renderer=CUDACompVolumeRenderer::Create(window_w,window_h);
    this->cuda_comp_volume_renderer->SetVolume(comp_volume);
}


