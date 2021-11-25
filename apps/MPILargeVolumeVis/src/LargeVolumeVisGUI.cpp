//
// Created by wyz on 2021/7/22.
//
#include <VolumeSlicer/Utils/plugin_loader.hpp>

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


#ifndef NDEBUG
#define SDL_EXPR(exec)                                                                                                 \
    exec;                                                                                                              \
    {                                                                                                                  \
        std::string __err_str = SDL_GetError();                                                                              \
        if (__err_str.length() > 0)                                                                                          \
        {                                                                                                              \
            LOG_ERROR("SDL call error: {0} in {1} function: {2} line: {3}",__err_str,__FILE__,__FUNCTION__,__LINE__);\
        }                                                                                                              \
    }

#define SDL_CHECK                                                                                                      \
    {                                                                                                                  \
        std::string __err_str = SDL_GetError();                                                                              \
        if (__err_str.length() > 0)                                                                                          \
        {                                                                                                              \
            LOG_ERROR("SDL call error: {0} in {1} before line: {2}",__err_str,__FILE__,__LINE__);\
        }                                                                                                               \
    }

#else
#define SDL_EXPR(exec) exec
#define SDL_CHECK
#endif


void LargeVolumeVisGUI::init(const char * config_file) {
    this->window_manager=std::make_unique<WindowManager>(config_file);
    this->window_w=window_manager->GetNodeWindowWidth();
    this->window_h=window_manager->GetNodeWindowHeight();
    window_manager->GetWorldVolumeSpace(volume_space_x,volume_space_y,volume_space_z);
    base_space=(std::min)({volume_space_x,volume_space_y,volume_space_z});

    SetCUDACtx(window_manager->GetGPUIndex());

//    PluginLoader::LoadPlugins("./plugins");

    initSDL();

    initRendererResource();
}

void LargeVolumeVisGUI::show() {
    bool exit=false;
    bool motion;

    auto process_event=[&exit,this,&motion](){
        static SDL_Event event;
        ImGui_ImplSDL2_ProcessEvent(&event);
        //camera pos according to volume dim count in voxel
        static control::FPSCamera fpsCamera({4.90f,5.858f,7.23f});
        static bool right_mouse_press;
        while(SDL_PollEvent(&event)){
            switch(event.type){
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
                        case SDLK_ESCAPE:{
                            exit=true;
                            auto pos=fpsCamera.getCameraPos();
                            LOG_ERROR("camera pos {0} {1} {2}",pos.x,pos.y,pos.z);
                            break;}
                        case SDLK_LCTRL:{
                            break;
                        }
                        case SDLK_a:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Left,0.0001);motion=true;break;}
                        case SDLK_d:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Right,0.0001);motion=true;break;}
                        case SDLK_w:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Forward,0.0001);motion=true;break;}
                        case SDLK_s:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Backward,0.0001);motion=true;break;}
                        case SDLK_q:{ fpsCamera.processKeyEvent(control::CameraDefinedKey::Up,0.0001);motion=true;break;}
                        case SDLK_e:{fpsCamera.processKeyEvent(control::CameraDefinedKey::Bottom,0.0001);motion=true;break;}
                        case SDLK_f:{
                            TransferFunc tf;
                            tf.points.emplace_back(0,std::array<double,4>{0.1,0.0,0.0,0.0});
                            tf.points.emplace_back(74,std::array<double,4>{0.0,0.0,0.0,0.0});
                            tf.points.emplace_back(127,std::array<double,4>{0.75,0.75,0.75,0.6});
                            tf.points.emplace_back(128,std::array<double,4>{1.0,0.3,1.0,1.0});
                            tf.points.emplace_back(255,std::array<double,4>{1.0,0.0,0.0,1.0});
                            this->comp_volume_renderer->SetTransferFunc(std::move(tf));
                        }
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
                    if(event.wheel.y>0){
                        fpsCamera.processMouseScroll(1.f);
                        motion=true;
                    }
                    else{
                        fpsCamera.processMouseScroll(-1.f);
                        motion=true;
                    }
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:{
                    if(event.button.button==1){
                        right_mouse_press=true;
                        fpsCamera.processMouseButton(control::CameraDefinedMouseButton::Left,true,event.button.x,event.button.y);
                        motion=true;
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
                        motion=true;
                    }
                    break;
                }
            }
        }//end of SDL_PollEvent
        MPI_Bcast(&fpsCamera,28,MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Bcast(&motion,1,MPI_INT,0,MPI_COMM_WORLD);
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
        comp_volume_renderer->SetMPIRender(mpiRenderParameter);
        Camera camera{};
        camera.pos={camera_pos.x,camera_pos.y,camera_pos.z};
        camera.up={camera_up.x,camera_up.y,camera_up.z};
        camera.look_at={camera_look_at.x,camera_look_at.y,camera_look_at.z};
        camera.right={camera_right.x,camera_right.y,camera_right.z};
        camera.zoom=camera_zoom;
        comp_volume_renderer->SetCamera(camera);
    };
    SDL_EXPR(sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED));
    SDL_Rect rect{0,0,(int)window_w,(int)window_h};
    SDL_PixelFormatEnum format=SDL_PIXELFORMAT_UNKNOWN;

    if(this->comp_volume_renderer->GetBackendName()=="opengl"){
        format = SDL_PIXELFORMAT_ABGR8888;
    }
    else if(this->comp_volume_renderer->GetBackendName()=="cuda"){
        format = SDL_PIXELFORMAT_ABGR8888;
    }
    else{
        format = SDL_PIXELFORMAT_RGBA8888;
    }
    SDL_Texture* texture=SDL_CreateTexture(sdl_renderer,format,SDL_TEXTUREACCESS_STREAMING,window_w,window_h);
    SDL_CHECK
    decltype(SDL_GetTicks()) cur_frame_t;
    uint32_t interval = window_manager->GetFrameTimeLock();
    bool flip = window_manager->GetRendererBackend() == "opengl";
    while(!exit){
        motion=false;
        cur_frame_t=SDL_GetTicks();
        process_event();

            comp_volume_renderer->resize(window_w,window_h);
            comp_volume_renderer->SetStep(base_space*0.5,6000);
            comp_volume_renderer->render(true);

            SDL_UpdateTexture(texture, NULL, comp_volume_renderer->GetImage().GetData(), window_w * 4);
            SDL_RenderClear(sdl_renderer);
            if(flip){
                SDL_RenderCopyEx(sdl_renderer,texture,&rect,&rect,0.0,nullptr,SDL_RendererFlip::SDL_FLIP_VERTICAL);
            }
            else{
                SDL_RenderCopy(sdl_renderer, texture, nullptr, &rect);
            }
            SDL_RenderPresent(sdl_renderer);

//        render_imgui();

        while(SDL_GetTicks()<cur_frame_t+interval){
        }
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
    SDL_EXPR(sdl_window = SDL_CreateWindow("MPILargeVolumeVis", window_manager->GetNodeScreenOffsetX(),window_manager->GetNodeScreenOffsetY() ,
                                           window_manager->GetNodeWindowWidth(), window_manager->GetNodeWindowHeight(), SDL_WINDOW_OPENGL|SDL_WINDOW_ALLOW_HIGHDPI));

    gl_context=SDL_GL_CreateContext(sdl_window);
    SDL_GL_MakeCurrent(sdl_window,gl_context);
    SDL_GL_SetSwapInterval(1);
    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)){
        LOG_CRITICAL("GLAD: OpenGL load failed.");
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
    this->comp_volume->SetSpaceX(volume_space_x);
    this->comp_volume->SetSpaceY(volume_space_y);
    this->comp_volume->SetSpaceZ(volume_space_z);

    if(this->window_manager->GetRendererBackend() == "cuda"){
        this->comp_volume_renderer=CUDACompVolumeRenderer::Create(window_w, window_h);
    }
    else if(this->window_manager->GetRendererBackend() == "opengl"){
        this->comp_volume_renderer=OpenGLCompVolumeRenderer::Create(window_w,window_h);
    }
    else{
        LOG_ERROR("Not supported renderer backend, use opengl as default");
        this->comp_volume_renderer=OpenGLCompVolumeRenderer::Create(window_w,window_h);
    }

    this->comp_volume_renderer->SetVolume(comp_volume);


    TransferFunc tf;
    tf.points.emplace_back(0,std::array<double,4>{0.1,0.0,0.0,0.0});
    tf.points.emplace_back(74,std::array<double,4>{0.0,0.0,0.0,0.0});
    tf.points.emplace_back(127,std::array<double,4>{0.8,0.7,0.2,0.6});
    tf.points.emplace_back(128,std::array<double,4>{0.8,0.7,0.2,1.0});
    tf.points.emplace_back(255,std::array<double,4>{1.0,0.2,0.1,1.0});
    this->comp_volume_renderer->SetTransferFunc(std::move(tf));

    CompRenderPolicy policy;
    policy.lod_dist[0]=0.3;
    policy.lod_dist[1]=0.6;
    policy.lod_dist[2]=1.2;
    policy.lod_dist[3]=2.4;
    policy.lod_dist[4]=4.8;
    policy.lod_dist[5]=9.6;
    policy.lod_dist[6]=std::numeric_limits<double>::max();
    policy.cdf_value_file="chebyshev_dist_mouse_cdf_config.json";
//    policy.volume_value_file="volume_value_mouse_cdf_config.json";
    this->comp_volume_renderer->SetRenderPolicy(policy);
}


