//
// Created by wyz on 2021/7/22.
//


#include "LargeVolumeVisGUI.hpp"
#include "camera.hpp"
#include "ShaderProgram.hpp"
#include "WindowManager.hpp"
#include "shader.hpp"

#include <VolumeSlicer/Utils/plugin_loader.hpp>
#include <VolumeSlicer/Utils/gl_helper.hpp>
#include <VolumeSlicer/render_helper.hpp>

#include <SDL.h>

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

    PluginLoader::LoadPlugins("./plugins");

    initSDL();

    initRendererResource();

}

void LargeVolumeVisGUI::show() {
    bool exit=false;

    auto process_event=[&exit,this](){
        static auto last_frame_t = SDL_GetTicks();
        auto current_frame_t = SDL_GetTicks();//ms
        auto delta_frame_t = current_frame_t-last_frame_t;
        last_frame_t = current_frame_t;
        float delta_t = delta_frame_t * 0.000001f;//s
        static SDL_Event event;

        //camera pos according to volume dim count in voxel
        std::array<float,3> view_pos;
        CompRenderHelper::GetDefaultViewPos(comp_volume,view_pos);
        static control::FPSCamera fpsCamera({view_pos[0],view_pos[1],view_pos[2]});
        static bool right_mouse_press;
        while(SDL_PollEvent(&event)){
            ImGui_ImplSDL2_ProcessEvent(&event);
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
                        case SDLK_a:{fpsCamera.processKeyEvent(control::CameraDefinedKey::MOVE_Left,delta_t);break;}
                        case SDLK_d:{fpsCamera.processKeyEvent(control::CameraDefinedKey::MOVE_Right,delta_t);break;}
                        case SDLK_w:{fpsCamera.processKeyEvent(control::CameraDefinedKey::MOVE_Forward,delta_t);break;}
                        case SDLK_s:{fpsCamera.processKeyEvent(control::CameraDefinedKey::MOVE_Backward,delta_t);break;}
                        case SDLK_q:{ fpsCamera.processKeyEvent(control::CameraDefinedKey::MOVE_Up,delta_t);break;}
                        case SDLK_e:{fpsCamera.processKeyEvent(control::CameraDefinedKey::MOVE_Bottom,delta_t);break;}
                        case SDLK_LEFT:{fpsCamera.processKeyEvent(control::CameraDefinedKey::ROTATE_Left,delta_t);break;}
                        case SDLK_RIGHT:{fpsCamera.processKeyEvent(control::CameraDefinedKey::ROTATE_Right,delta_t);break;}
                        case SDLK_UP:{fpsCamera.processKeyEvent(control::CameraDefinedKey::ROTATE_Up,delta_t);break;}
                        case SDLK_DOWN:{fpsCamera.processKeyEvent(control::CameraDefinedKey::ROTATE_Down,delta_t);break;}
                        case SDLK_r:{fpsCamera.processKeyEvent(control::CameraDefinedKey::ZOOM_IN,delta_t);break;}
                        case SDLK_t:{fpsCamera.processKeyEvent(control::CameraDefinedKey::ZOOM_Out,delta_t);break;}
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
        }//end of SDL_PollEvent
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
//        comp_volume_renderer->SetMPIRender(mpiRenderParameter);
        Camera camera{};
        camera.pos={camera_pos.x,camera_pos.y,camera_pos.z};
        camera.up={camera_up.x,camera_up.y,camera_up.z};
        camera.look_at={camera_look_at.x,camera_look_at.y,camera_look_at.z};
        camera.right={camera_right.x,camera_right.y,camera_right.z};
        camera.zoom=camera_zoom;
        comp_volume_renderer->SetCamera(camera);
    };

    SDL_CHECK
    decltype(SDL_GetTicks()) cur_frame_t;
    uint32_t interval = window_manager->GetFrameTimeLock();


    bool isOpenGL = comp_volume_renderer->GetBackendName() == "opengl";

    //for cuda backend renderer
    if(!isOpenGL){
        SDL_EXPR(SDL_GL_MakeCurrent(sdl_window,gl_context));
        createScreenQuad();
        createGLTexture();
        comp_render_shader = std::make_unique<Shader>();
        comp_render_shader->setShader(shader::composite_render_v,shader::conposite_render_f);
    }
    float step;
    int steps;
    if(window_manager->GetAdvanceOptions().ray_cast.step.has_value() && window_manager->GetAdvanceOptions().ray_cast.steps.has_value()){
        step = window_manager->GetAdvanceOptions().ray_cast.step.value();
        steps = window_manager->GetAdvanceOptions().ray_cast.steps.value();
    }
    else{
        CompRenderHelper::GetDefaultRayCastStep(comp_volume,step,steps);
    }
    while(!exit){
        cur_frame_t=SDL_GetTicks();
        process_event();

        comp_volume_renderer->resize(window_w,window_h);
        comp_volume_renderer->SetStep(step,steps);
        comp_volume_renderer->render(true);

        if(!isOpenGL){
            glClearColor(0.f,0.f,0.f,1.f);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            glTextureSubImage2D(comp_render_tex,0,0,0,window_w,window_h,GL_RGBA,GL_UNSIGNED_BYTE,
                                comp_volume_renderer->GetImage().GetData());
            comp_render_shader->use();
            comp_render_shader->setInt("height",window_h);
            glBindVertexArray(screen_quad_vao);
            glDrawArrays(GL_TRIANGLES,0,6);
            GL_CHECK
        }
        //todo image based post-process
        {

        }

        if(window_manager->IsRoot()){
            render_imgui();
        }


        while(SDL_GetTicks()<cur_frame_t+interval){

        }
        SDL_GL_SwapWindow(sdl_window);

        SDL_CHECK
    }

}
void LargeVolumeVisGUI::render_imgui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(sdl_window);
    ImGui::NewFrame();
    {
        ImGui::Begin("MPI-LargeVolume-Vis-Info");
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
        LOG_CRITICAL("{0} - SDL could not initialize! SDL Error: {1}.", __FUNCTION__, SDL_GetError());
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

    SDL_EXPR(gl_context=SDL_GL_CreateContext(sdl_window));
    SDL_EXPR(SDL_GL_MakeCurrent(sdl_window,gl_context));
    SDL_GL_SetSwapInterval(1);
    if(!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)){
        LOG_CRITICAL("GLAD: OpenGL load failed.");
        throw std::runtime_error("GLAD: OpenGL load failed.");
    }
    glEnable(GL_DEPTH_TEST);
    SDL_CHECK
    assert(sdl_window && gl_context);
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
    //destroy resources first
    {
        comp_volume.reset();
        comp_volume_renderer.reset(nullptr);
    }
    if(!gl_context || !sdl_window) return;
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
    }
    SDL_GL_DeleteContext(gl_context);
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
        this->comp_volume_renderer=OpenGLCompVolumeRenderer::Create(window_w,window_h,false);
    }
    else{
        LOG_ERROR("Not supported renderer backend, use opengl as default");
        this->comp_volume_renderer=OpenGLCompVolumeRenderer::Create(window_w,window_h);
    }

    this->comp_volume_renderer->SetVolume(comp_volume);


    const auto& tf_map = window_manager->GetTFMap();
    TransferFunc tf;
    for(const auto& item:tf_map){
        tf.points.emplace_back(item.first,item.second);
    }
    this->comp_volume_renderer->SetTransferFunc(std::move(tf));

    std::vector<double> lod_dist;
    if(window_manager->GetAdvanceOptions().lod_policy.lod_dist.has_value()){
        lod_dist = window_manager->GetAdvanceOptions().lod_policy.lod_dist.value();
    }
    else{
        CompRenderHelper::GetBaseLodPolicy(comp_volume,lod_dist);
    }
//    assert(lod_dist.size() == sizeof(CompRenderPolicy::lod_dist)/sizeof(std::remove_extent<decltype(CompRenderPolicy::lod_dist)>));
    LOG_INFO("size {} {}.",sizeof(CompRenderPolicy::lod_dist),sizeof(std::remove_extent<decltype(CompRenderPolicy::lod_dist)>));
    CompRenderPolicy policy;
    std::copy(lod_dist.begin(),lod_dist.end(),policy.lod_dist);

//    policy.cdf_value_file="chebyshev_dist_mouse_cdf_config.json";
//    policy.volume_value_file="volume_value_mouse_cdf_config.json";
    this->comp_volume_renderer->SetRenderPolicy(policy);

}

void LargeVolumeVisGUI::createScreenQuad()
{
    static std::array<float,24> screen_quad_vertices={
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
        1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
        1.0f, -1.0f,  1.0f, 0.0f,
        1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1,&screen_quad_vao);
    glGenBuffers(1,&screen_quad_vbo);
    glBindVertexArray(screen_quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER,screen_quad_vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(screen_quad_vertices),screen_quad_vertices.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    GL_CHECK
}

void LargeVolumeVisGUI::createGLTexture()
{
    glGenTextures(1,&comp_render_tex);
    glBindTexture(GL_TEXTURE_2D,comp_render_tex);
    glBindTextureUnit(0,comp_render_tex);
    glTextureStorage2D(comp_render_tex,1,GL_RGBA8,window_w,window_h);
    glBindImageTexture(0,comp_render_tex,0,GL_FALSE,0,GL_READ_WRITE,GL_RGBA8);

    GL_CHECK
}
