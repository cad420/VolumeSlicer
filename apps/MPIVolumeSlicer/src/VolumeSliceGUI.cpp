//
// Created by wyz on 2021/7/12.
//



#include "VolumeSliceGUI.hpp"
#include "WindowManager.hpp"
#include "ShaderProgram.hpp"
#include <glad/glad.h>
#include <SDL.h>
#include <spdlog/spdlog.h>
#include <vector>


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
VolumeSliceGUI::VolumeSliceGUI()
:window_w(0),window_h(0),volume_space_x(1.f),volume_space_y(1.f),volume_space_z(1.f)
{


}

VolumeSliceGUI::~VolumeSliceGUI() {
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
void VolumeSliceGUI::init(const char *config_file) {
    //1. window manager
    this->window_manager=std::make_unique<WindowManager>(config_file);
    this->window_w=window_manager->GetNodeWindowWidth();
    this->window_h=window_manager->GetNodeWindowHeight();
    window_manager->GetWorldVolumeSpace(volume_space_x,volume_space_y,volume_space_z);
    SetCUDACtx(0);
    //2. init SDL
    initSDL();
    //3. init OpenGL resource
    initGLResource();
    //4. set comp volume
    set_comp_volume(window_manager->GetNodeResourcePath().c_str());
    //5. set raw volume
    if(window_manager->IsRoot()){
        window_manager->GetRawDim(raw_dim_x,raw_dim_y,raw_dim_z);
        set_raw_volume(window_manager->GetRawResourcePath(this->raw_lod).c_str(),
                       raw_dim_x,raw_dim_y,raw_dim_z);
    }
}
void VolumeSliceGUI::initSDL() {
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
    SDL_EXPR(sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED));
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

void operator+=(std::array<float,4>& a1,const std::array<float,4>& a2){
    a1[0]+=a2[0];
    a1[1]+=a2[1];
    a1[2]+=a2[2];
    a1[3]+=a2[3];
}
std::array<float,4> operator*(const std::array<float,4>& a,float t){
    return {a[0]*t,a[1]*t,a[2]*t,a[3]*t};
}
std::array<float,4> operator-(const std::array<float,4>& a1,const std::array<float,4>& a2){
    return {a1[0]-a2[0],a1[1]-a2[1],a1[2]-a2[2],a1[3]-a2[3]};
}
std::array<float,4> operator+(const std::array<float,4>& a1,const std::array<float,4>& a2){
    return {a1[0]+a2[0],a1[1]+a2[1],a1[2]+a2[2],a1[3]+a2[3]};
}
std::array<float,4> operator*(const std::array<float,4>& a1,const std::array<float,4>& a2){
    return {a1[0]*a2[0],a1[1]*a2[1],a1[2]*a2[2],a1[3]*a2[3]};
}
void VolumeSliceGUI::show() {
    spdlog::set_level(spdlog::level::info);
    bool exit=false;
    auto process_event=[&exit,this](){
        static SDL_Event event;
        ImGui_ImplSDL2_ProcessEvent(&event);
        static bool scaling=false;
        static bool focus=false;
        static bool left_mouse_button_pressed;
        static constexpr float one_degree=1.0/180.0*3.141592627;
        while(SDL_PollEvent(&event)){
            switch(event.type){
                case SDL_QUIT:{
                    exit=true;
                    break;
                }
                case SDL_WINDOWEVENT:{
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_FOCUS_GAINED:focus=true;break;
                        case SDL_WINDOWEVENT_FOCUS_LOST:focus=false;break;
                    }
                    break;
                }
                case SDL_KEYDOWN:{
                    switch (event.key.keysym.sym) {
                        case SDLK_ESCAPE:{exit=true;break;}
                        case SDLK_LCTRL:{scaling=true;break;}
                        case SDLK_a:{world_slicer->RotateByX(one_degree);break;}
                        case SDLK_d:{world_slicer->RotateByX(-one_degree);break;}
                        case SDLK_w:{world_slicer->RotateByY(one_degree);break;}
                        case SDLK_s:{world_slicer->RotateByY(-one_degree);break;}
                        case SDLK_q:{world_slicer->RotateByZ(one_degree);break;}
                        case SDLK_e:{world_slicer->RotateByZ(-one_degree);break;}
                        case SDLK_m:{show_imgui = !show_imgui;break;}
                        case SDLK_LEFT:
                        case SDLK_DOWN:{
                            auto lod=world_slicer->GetSlice().voxel_per_pixel_width;
                            world_slicer->MoveByNormal(-lod);
                            break;
                        }
                        case SDLK_RIGHT:
                        case SDLK_UP:{
                            auto lod=world_slicer->GetSlice().voxel_per_pixel_width;
                            world_slicer->MoveByNormal(lod);
                            break;
                        }
                    }
                    break;
                }
                case SDL_KEYUP:{
                    switch (event.key.keysym.sym) {
                        case SDLK_LCTRL:{
                            scaling=false;
                            break;
                        }
                    }
                    break;
                }
                case SDL_MOUSEWHEEL:{
                    if(!focus) break;
                    if(scaling){
                        if(event.wheel.y>0){
                            world_slicer->StretchInXY(1.1f,1.1f);
                        }
                        else{
                            world_slicer->StretchInXY(0.9f,0.9f);
                        }
                    }
                    else{
                        auto lod=world_slicer->GetSlice().voxel_per_pixel_width;
                        if(event.wheel.y>0)
                            world_slicer->MoveByNormal(lod);
                        else
                            world_slicer->MoveByNormal(-lod);
                    }
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:{

                    if(event.button.button==1){
                        left_mouse_button_pressed=true;
                    }
                    break;
                }
                case SDL_MOUSEBUTTONUP:{

                    if(event.button.button==1){
                        left_mouse_button_pressed=false;
                    }
                    break;
                }
                case SDL_MOUSEMOTION:{
                    if(left_mouse_button_pressed){
                        world_slicer->MoveInPlane(-event.motion.xrel,-event.motion.yrel);
                    }
                    break;
                }
            }
        }//end of SDL_PollEvent

        if(window_manager->GetWindowRank()==0){
            world_slice=world_slicer->GetSlice();
        }
//        std::cout<<"before bcast"<<std::endl;
        MPI_Bcast(&world_slice,20,MPI_FLOAT,0,MPI_COMM_WORLD);
//        MPI_Barrier(MPI_COMM_WORLD);
//        std::cout<<"node "<<window_manager->GetWindowRank()<<std::endl;
//        std::cout<<world_slice.voxel_per_pixel_width<<" "<<world_slice.voxel_per_pixel_height<<std::endl;
        auto slice=world_slice;
        slice.n_pixels_width=window_manager->GetNodeWindowWidth();
        slice.n_pixels_height=window_manager->GetNodeWindowHeight();
        float center_x=window_manager->GetWindowColNum()*1.f/2-0.5f;
        float center_y=window_manager->GetWindowRowNum()*1.f/2-0.5f;
        float base_space=std::min({volume_space_x,volume_space_y,volume_space_z});
        std::array<float,4> t={base_space/volume_space_x,base_space/volume_space_y,base_space/volume_space_z};
        slice.origin+=
                (slice.right * ((window_manager->GetWorldRankOffsetX()-center_x)*slice.n_pixels_width*slice.voxel_per_pixel_width)
                +slice.up*((window_manager->GetWorldRankOffsetY()-center_y)*slice.n_pixels_height*slice.voxel_per_pixel_height))*t;

        this->node_slice = slice;
        MPI_Barrier(MPI_COMM_WORLD);
    };
    constexpr uint32_t frame_time=1000/50;
    uint32_t last_frame_time=SDL_GetTicks();


    GL_CHECK
    while(!exit){
        process_event();

        glClearColor(0.f,0.f,0.f,0.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
//        SDL_UpdateTexture(texture, NULL, image.data(), window_w*sizeof(uint32_t));
//        SDL_RenderClear(sdl_renderer);
//        SDL_RenderCopy(sdl_renderer, texture, nullptr, nullptr);
//        SDL_RenderPresent(sdl_renderer);

        GL_CHECK
        if(window_manager->IsRoot()){
            render_root_frame();
        }
        else{
            render_node_frame();
        }



        if(SDL_GetTicks()-last_frame_time<frame_time){
            SDL_Delay(frame_time-SDL_GetTicks()+last_frame_time);
        }
        last_frame_time=SDL_GetTicks();

        SDL_GL_SwapWindow(sdl_window);
        SDL_CHECK
    }

}

void VolumeSliceGUI::set_comp_volume(const char *file) {
    this->comp_volume=CompVolume::Load(file);
    comp_volume->SetSpaceX(volume_space_x);
    comp_volume->SetSpaceY(volume_space_y);
    comp_volume->SetSpaceZ(volume_space_z);
    Slice slice;
    slice.n_pixels_width=window_manager->GetWorldWindowWidth();
    slice.n_pixels_height=window_manager->GetWorldWindowHeight();
    slice.voxel_per_pixel_height=slice.voxel_per_pixel_width=2.f;
    slice.origin={comp_volume->GetVolumeDimX()/2.f,
                  comp_volume->GetVolumeDimY()/2.f,
                  comp_volume->GetVolumeDimZ()/2.f};
    slice.normal={0.f,0.f,1.f,0.f};
    slice.right={1.f,0.f,0.f,0.f};
    slice.up={0.f,1.f,0.f,0.f};

    this->world_slicer=Slicer::CreateSlicer(slice);

    comp_sample_frame.width=window_w;
    comp_sample_frame.height=window_h;
    comp_sample_frame.channels=1;
    comp_sample_frame.data.resize((size_t)window_w*window_h,0);
    this->comp_volume_sampler=VolumeSampler::CreateVolumeSampler(comp_volume);
}

void VolumeSliceGUI::set_raw_volume(const char *file,uint32_t dim_x,uint32_t dim_y,uint32_t dim_z) {
    this->raw_volume=RawVolume::Load(file,vs::VoxelType::UInt8,{dim_x,dim_y,dim_z},
                                     {volume_space_x,
                                      volume_space_y,
                                      volume_space_z});
    this->raw_volume_sampler=VolumeSampler::CreateVolumeSampler(raw_volume);

    raw_sample_frame.width=window_w;
    raw_sample_frame.height=window_h;
    raw_sample_frame.channels=1;
    raw_sample_frame.data.resize((size_t)window_w*window_h,0);

//    this->volume_slice_renderer=CreateRenderer(window_w,window_h);
//    volume_slice_renderer->SetVolume(raw_volume);
}

void VolumeSliceGUI::render_imgui() {

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(sdl_window);
    ImGui::NewFrame();
    {
        ImGui::Begin("VolumeSlice");
        ImGui::Text("FPS: %.1f",ImGui::GetIO().Framerate);
        ImGui::InputFloat("Space X",&volume_space_x,0.0f,0.0f,"%.5f");
        ImGui::InputFloat("Space Y",&volume_space_y,0.0f,0.0f,"%.5f");
        ImGui::InputFloat("Space Z",&volume_space_z,0.0f,0.0f,"%.5f");
        ImGui::RadioButton("Default",&raw_mode,0);
        ImGui::RadioButton("Zoom Slice",&raw_mode,1);
//        ImGui::RadioButton("Volume Slice",&raw_mode,2);
        this->comp_volume->SetSpaceX(volume_space_x);
        this->comp_volume->SetSpaceY(volume_space_y);
        this->comp_volume->SetSpaceZ(volume_space_z);
        float base_space=std::min({volume_space_x,volume_space_y,volume_space_z});
        this->world_slicer->SetSliceSpaceRatio({volume_space_x/base_space,
                                                volume_space_y/base_space,
                                                volume_space_z/base_space});
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

void VolumeSliceGUI::render_root_frame() {

    switch (raw_mode)
    {
        case 0:{
            render_node_frame();
            break;
        }
        case 1:{
            render_zoom_slice();
            break;
        }
        case 2:{
            render_volume_slice();
            break;
        }
    }
    if(show_imgui){
        render_imgui();
    }

}
void VolumeSliceGUI::render_zoom_slice()
{
    auto slice = this->world_slicer->GetSlice();
    static float ratio = pow(2,raw_lod);
    slice.origin={slice.origin[0]/ratio,slice.origin[1]/ratio,slice.origin[2]/ratio,1.f};
    static float base_space=(std::min)({volume_space_x,volume_space_y,volume_space_z});
    static float space_ratio_x=volume_space_x/base_space;
    static float space_ratio_y=volume_space_y/base_space;
    static float space_ratio_z=volume_space_z/base_space;
    float A = slice.normal[0]*space_ratio_x;
    float B = slice.normal[1]*space_ratio_y;
    float C = slice.normal[2]*space_ratio_z;
    float length = std::sqrt(A*A+B*B+C*C);
    A /= length;
    B /= length;
    C /= length;
    float x0 = slice.origin[0];
    float y0 = slice.origin[1];
    float z0 = slice.origin[2];
    float D  = A*x0 + B*y0 + C*z0;
    static std::array<std::array<float,3>,8> pts={
        std::array<float,3>{0.0f,0.0f,0.0f},
        std::array<float,3>{raw_dim_x*1.f,0.f,0.f},
        std::array<float,3>{raw_dim_x*1.f,1.f*raw_dim_y,0.f},
        std::array<float,3>{0.f,1.f*raw_dim_y,0.f},
        std::array<float,3>{0.f,0.f,1.f*raw_dim_z},
        std::array<float,3>{1.f*raw_dim_x,0.f,1.f*raw_dim_z},
        std::array<float,3>{1.f*raw_dim_x,1.f*raw_dim_y,1.f*raw_dim_z},
        std::array<float,3>{0.f,1.f*raw_dim_y,1.f*raw_dim_z}
    };
    static std::array<std::array<int,2>,12> line_index={
        std::array<int,2>{0,1},
        std::array<int,2>{1,2},
        std::array<int,2>{2,3},
        std::array<int,2>{3,0},
        std::array<int,2>{4,5},
        std::array<int,2>{5,6},
        std::array<int,2>{6,7},
        std::array<int,2>{7,4},
        std::array<int,2>{0,4},
        std::array<int,2>{1,5},
        std::array<int,2>{2,6},
        std::array<int,2>{3,7}
    };
    int intersect_pts_cnt=0;
    float t,k;
    std::array<float,3> intersect_pts={0.f,0.f,0.f};
    std::array<float,3> tmp;
    float x1,y1,z1,x2,y2,z2;
    for(int i=0;i<line_index.size();i++){
        x1=pts[line_index[i][0]][0];
        y1=pts[line_index[i][0]][1];
        z1=pts[line_index[i][0]][2];
        x2=pts[line_index[i][1]][0];
        y2=pts[line_index[i][1]][1];
        z2=pts[line_index[i][1]][2];
        k=A*(x1-x2)+B*(y1-y2)+C*(z1-z2);
        if(std::abs(k)>0.0001f){
            t=( D - (A*x2+B*y2+C*z2) ) / k;
            if(t>=0.f && t<1.f){
                intersect_pts_cnt++;
                tmp={t*x1+(1-t)*x2,t*y1+(1-t)*y2,t*z1+(1-t)*z2};
                intersect_pts={intersect_pts[0]+tmp[0],
                               intersect_pts[1]+tmp[1],
                               intersect_pts[2]+tmp[2]};
            }
        }
    }
    intersect_pts={intersect_pts[0]/intersect_pts_cnt,
                   intersect_pts[1]/intersect_pts_cnt,
                   intersect_pts[2]/intersect_pts_cnt};
    slice.origin={intersect_pts[0],
                  intersect_pts[1],
                  intersect_pts[2],1.f};
    slice.voxel_per_pixel_width=slice.voxel_per_pixel_height=0.8f;
    slice.n_pixels_width=window_w;
    slice.n_pixels_height=window_h;

    GL_CHECK
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    raw_volume_sampler->Sample(slice,raw_sample_frame.data.data());



    auto world_slice = world_slicer->GetSlice();
    float p =world_slice.voxel_per_pixel_height/ratio;
    glm::vec3 right={world_slice.right[0],world_slice.right[1],world_slice.right[2]};
    glm::vec3 up={world_slice.up[0],world_slice.up[1],world_slice.up[2]};
//    float x_t=std::abs(glm::dot(right,{1,1,3}));
//    float y_t=std::abs(glm::dot(up,{1,1,3}));
    glm::vec3 offset={(world_slice.origin[0]/ratio-slice.origin[0])*space_ratio_x,
                      (world_slice.origin[1]/ratio-slice.origin[1])*space_ratio_y,
                      (world_slice.origin[2]/ratio-slice.origin[2])*space_ratio_z};
    float x_offset=glm::dot(right,offset);
    float y_offset=-glm::dot(up,offset);
    int min_p_x=x_offset/slice.voxel_per_pixel_width
                + slice.n_pixels_width/2 - world_slice.n_pixels_width/2*p/slice.voxel_per_pixel_width;
    int min_p_y=y_offset/slice.voxel_per_pixel_height
                + slice.n_pixels_height/2 - world_slice.n_pixels_height/2*p/slice.voxel_per_pixel_width;
    int max_p_x=x_offset/slice.voxel_per_pixel_width
                + slice.n_pixels_width/2 + world_slice.n_pixels_width/2*p/slice.voxel_per_pixel_width;
    int max_p_y=y_offset/slice.voxel_per_pixel_height
                + slice.n_pixels_height/2 + world_slice.n_pixels_height/2*p/slice.voxel_per_pixel_width;
    min_p_x=min_p_x<0?0:min_p_x;
    max_p_x=max_p_x<slice.n_pixels_width?max_p_x:slice.n_pixels_width-1;
    min_p_y=min_p_y<0?0:min_p_y;
    max_p_y=max_p_y<slice.n_pixels_height?max_p_y:slice.n_pixels_height-1;


    glTextureSubImage2D(comp_sample_tex,0,0,0,window_w,window_h,GL_RED,GL_UNSIGNED_BYTE,raw_sample_frame.data.data());
    glBindTextureUnit(0,comp_sample_tex);
    GL_CHECK
    GL_EXPR(zoom_slice_render_shader->use());
    GL_EXPR(zoom_slice_render_shader->setInt("min_p_x",min_p_x));
    GL_EXPR(zoom_slice_render_shader->setInt("min_p_y",min_p_y));
    GL_EXPR(zoom_slice_render_shader->setInt("max_p_x",max_p_x));
    GL_EXPR(zoom_slice_render_shader->setInt("max_p_y",max_p_y));
    GL_CHECK
    glBindVertexArray(screen_quad_vao);
    GL_CHECK
    glDrawArrays(GL_TRIANGLES,0,6);
    GL_CHECK
}
void VolumeSliceGUI::render_volume_slice()
{

}
void VolumeSliceGUI::render_node_frame() {
    GL_CHECK
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    comp_volume_sampler->Sample(node_slice,comp_sample_frame.data.data());

    glTextureSubImage2D(comp_sample_tex,0,0,0,window_w,window_h,GL_RED,GL_UNSIGNED_BYTE,comp_sample_frame.data.data());
    glBindTextureUnit(0,comp_sample_tex);
    GL_CHECK
    comp_render_shader->use();
    GL_CHECK
    glBindVertexArray(screen_quad_vao);
    GL_CHECK
    glDrawArrays(GL_TRIANGLES,0,6);
    GL_CHECK
}

void VolumeSliceGUI::createScreenQuad() {
    std::array<float,24> screen_quad_vertices={
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

void VolumeSliceGUI::initGLResource() {
    createShader();
    createScreenQuad();
    createGLTexture();
}

void VolumeSliceGUI::createShader() {
    comp_render_shader=std::make_unique<Shader>(
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\MPIVolumeSlicer\\src\\shader\\comp_render_v.glsl",
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\MPIVolumeSlicer\\src\\shader\\comp_render_f.glsl"
            );
    zoom_slice_render_shader=std::make_unique<Shader>(
        "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\MPIVolumeSlicer\\src\\shader\\zoom_slice_render_v.glsl",
        "C:\\Users\\wyz\\projects\\VolumeSlicer\\apps\\MPIVolumeSlicer\\src\\shader\\zoom_slice_render_f.glsl"
    );
}

void VolumeSliceGUI::createGLTexture() {
    glGenTextures(1,&comp_sample_tex);
    glBindTexture(GL_TEXTURE_2D,comp_sample_tex);
    glTextureStorage2D(comp_sample_tex,1,GL_R8,window_w,window_h);
    glBindImageTexture(0,comp_sample_tex,0,GL_FALSE,0,GL_READ_ONLY,GL_R8);

    glCreateTextures(GL_TEXTURE_2D,1,&raw_sample_tex);
    glTextureStorage2D(raw_sample_tex,1,GL_R8,window_w/2,window_h);
    glBindImageTexture(1,raw_sample_tex,0,GL_FALSE,0,GL_READ_ONLY,GL_R8);

    glCreateTextures(GL_TEXTURE_2D,1,&raw_render_tex);
    glTextureStorage2D(raw_render_tex,1,GL_RGBA8,window_w/2,window_h);
    glBindImageTexture(2,raw_render_tex,0,GL_FALSE,0,GL_READ_ONLY,GL_RGBA8);

    GL_CHECK
}

