//
// Created by wyz on 2021/11/5.
//
#include "LargeMeshVisGUI.hpp"
#include "camera.hpp"
#include "WindowManager.hpp"
#include <SDL.h>
#include <glad/glad.h>
#include <VolumeSlicer/Utils/gl_helper.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/timer.hpp>
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

LargeMeshVisGUI::LargeMeshVisGUI()
{


}

LargeMeshVisGUI::~LargeMeshVisGUI()
{
    SDL_EXPR(SDL_DestroyRenderer(sdl_renderer));
    SDL_EXPR(SDL_DestroyWindow(sdl_window));
    SDL_EXPR(SDL_Quit());
}

void LargeMeshVisGUI::init(const char *config_file)
{
    this->window_manager = std::make_unique<WindowManager>(config_file);
    this->window_w = window_manager->GetNodeWindowWidth();
    this->window_h = window_manager->GetNodeWindowHeight();
    window_manager->GetWorldVolumeSpace(space_x,space_y,space_z);
    base_space = (std::min)({space_x,space_y,space_z});

    initSDL();

    initResource();
}

void LargeMeshVisGUI::show()
{
    bool exit = false;
        auto process_event=[&exit,this](){
        static SDL_Event event;
        static control::FPSCamera fpsCamera({1.12f,0.94f,4.0f});
        static bool right_mouse_press;
        static bool show_mouse = false;
        while(SDL_PollEvent(&event)){
            switch (event.type)
            {
                case SDL_QUIT:{
                    exit=true;
                    auto pos=fpsCamera.getCameraPos();
                    LOG_ERROR("camera pos {0} {1} {2}",pos.x,pos.y,pos.z);
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
                        case SDLK_ESCAPE:{exit=true;
                          auto pos=fpsCamera.getCameraPos();
                          LOG_ERROR("camera pos {0} {1} {2}",pos.x,pos.y,pos.z);
                          break;}
                        case SDLK_LCTRL:{
                          break;
                        }
                        case SDLK_a:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Left,0.0001);break;}
                        case SDLK_d:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Right,0.0001);break;}
                        case SDLK_w:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Forward,0.0001);break;}
                        case SDLK_s:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Backward,0.0001);break;}
                        case SDLK_q:{ fpsCamera.processKeyEvent(control::CameraDefinedKey::Up,0.0001);break;}
                        case SDLK_e:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Bottom,0.0001);break;}
                        case SDLK_h:{show_mouse = !show_mouse;SDL_ShowCursor(show_mouse);break;}
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
                case SDL_MOUSEWHEEL:{
                    if(event.wheel.y>0){
                      fpsCamera.processMouseScroll(1.f);
                    }
                    else{
                      fpsCamera.processMouseScroll(-1.f);
                    }
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:{

                    if(event.button.button==1){
                      right_mouse_press=true;
                      fpsCamera.processMouseButton(control::CameraDefinedMouseButton::Left,true,event.button.x,event.button.y);
                    }
                    break;
                }
                case SDL_MOUSEBUTTONUP:{

                    if(event.button.button==1){
                      right_mouse_press=false;
                      fpsCamera.processMouseButton(control::CameraDefinedMouseButton::Left,false,event.button.x,event.button.y);

                    }
                    break;
                }
                case SDL_MOUSEMOTION:{
                    if(right_mouse_press){
                      fpsCamera.processMouseMove(event.button.x,event.button.y);
                    }
                    break;
                }
            }
        }
          MPI_Bcast(&fpsCamera,28,MPI_FLOAT,0,MPI_COMM_WORLD);
          MPI_Bcast(&exit,1,MPI_INT,0,MPI_COMM_WORLD);
          MPI_Barrier(MPI_COMM_WORLD);
          auto camera_pos=fpsCamera.getCameraPos();
          auto camera_up=fpsCamera.getCameraUp();
          auto camera_look_at=fpsCamera.getCameraLookAt();
          auto camera_zoom=fpsCamera.getZoom();
          auto camera_right=fpsCamera.getCameraRight();
          MPIRenderParameter mpiRenderParameter;
          mpiRenderParameter.mpi_world_window_w=window_manager->GetWorldWindowWidth();
          mpiRenderParameter.mpi_world_window_h=window_manager->GetWorldWindowHeight();
          static float center_x=window_manager->GetWindowColNum()*1.f/2-0.5f;
          static float center_y=window_manager->GetWindowRowNum()*1.f/2-0.5f;
          mpiRenderParameter.mpi_node_x_offset=(window_manager->GetWorldRankOffsetX()-center_x);
          mpiRenderParameter.mpi_node_y_offset=(window_manager->GetWorldRankOffsetY()-center_y);
          mpiRenderParameter.mpi_world_row_num=window_manager->GetWindowRowNum();
          mpiRenderParameter.mpi_world_col_num=window_manager->GetWindowColNum();
          mpiRenderParameter.mpi_node_x_index=window_manager->GetWorldRankOffsetX();
          mpiRenderParameter.mpi_node_y_index=window_manager->GetWorldRankOffsetY();
          mesh_renderer->SetMPIRender(mpiRenderParameter);
          Camera camera{};
          camera.pos={camera_pos.x,camera_pos.y,camera_pos.z};
          camera.up={camera_up.x,camera_up.y,camera_up.z};
          camera.look_at={camera_look_at.x,camera_look_at.y,camera_look_at.z};
          camera.right={camera_right.x,camera_right.y,camera_right.z};
          camera.zoom=camera_zoom;
          mesh_renderer->SetCamera(camera);
    };
    SDL_EXPR(sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED));
    SDL_Rect rect{0,0,(int)window_w,(int)window_h};
    SDL_Texture* texture=SDL_CreateTexture(sdl_renderer,SDL_PIXELFORMAT_ABGR8888,SDL_TEXTUREACCESS_STREAMING,window_w,window_h);
    while(!exit){
        AutoTimer timer;
        process_event();

        this->mesh_renderer->render();

        SDL_UpdateTexture(texture, NULL, mesh_renderer->GetImage().GetData(), window_w * 4);

        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopyEx(sdl_renderer, texture, nullptr, &rect, 0, NULL, SDL_FLIP_VERTICAL);

        SDL_RenderPresent(sdl_renderer);
    }

}

void LargeMeshVisGUI::initSDL()
{
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
    SDL_EXPR(sdl_window = SDL_CreateWindow("MPILargeMeshVis", window_manager->GetNodeScreenOffsetX(),window_manager->GetNodeScreenOffsetY() ,
                                           window_manager->GetNodeWindowWidth(), window_manager->GetNodeWindowHeight(), SDL_WINDOW_OPENGL|SDL_WINDOW_ALLOW_HIGHDPI));
    SDL_ShowCursor(false);
}

void LargeMeshVisGUI::initResource()
{
    this->mesh = Mesh::Load(window_manager->GetNodeResourcePath());
    this->mesh->Transform(space_x,space_y,space_z);
    this->mesh_renderer = SimpleMeshRenderer::Create(window_w,window_h);
    this->mesh_renderer->SetMesh(this->mesh);
}
