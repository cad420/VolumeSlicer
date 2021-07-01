//
// Created by wyz on 2021/6/8.
//

#include<random>
#include<glm/gtc/matrix_transform.hpp>
#include"Common/gl_helper.hpp"
#include"Render/render_impl.hpp"
#include"Render/transfer_function_impl.hpp"
#include"Render/wgl_wrap.hpp"
#define WGL_NV_gpu_affinity


VS_START


std::unique_ptr<RawVolumeRenderer> CreateRenderer(int w, int h) {
    return std::make_unique<MultiVolumeRender>(w,h);
}

MultiVolumeRender::MultiVolumeRender(int w, int h) {
    if(w>0 && h>0 && w<MAX_SLICE_W && h<MAX_SLICE_H) {
        this->window_width=w;
        this->window_height=h;
        volume_tex=0;
        volume_x=volume_y=volume_z=0;
        space_x=space_y=space_z=0.f;
        volume_visible=slice_visible=false;
        x0=y0=z0=0.f;
        x1=y1=z1=1.f;
        volume_board_vao=volume_board_vbo=volume_board_ebo=0;
        volume_visible_board_vao=volume_visible_board_vbo=volume_visible_board_ebo=0;
        transfer_func_tex=preInt_tf_tex=0;
        slice_vao=slice_vbo=0;
        screen_quad_vao=screen_quad_vbo=0;
        initGL();
        setPosFrameBuffer();
        setScreenQuad();
        setShader();
    }
}


void MultiVolumeRender::SetVolume(const std::shared_ptr<RawVolume>& volume) noexcept {
    if(!volume){
        spdlog::error("Set empty volume.");
        return;
    }
    this->volume_visible=true;

    this->volume_x=volume->GetVolumeDimX();
    this->volume_y=volume->GetVolumeDimY();
    this->volume_z=volume->GetVolumeDimZ();
    this->space_x=volume->GetVolumeSpaceX();
    this->space_y=volume->GetVolumeSpaceY();
    this->space_z=volume->GetVolumeSpaceZ();

    setVolumeBoard();
    setVisibleBoard();

    if(volume_tex){
        //delete old volume texture
        glDeleteTextures(1,&volume_tex);
    }
    glGenTextures(1,&volume_tex);
    glBindTexture(GL_TEXTURE_3D,volume_tex);
//    glBindTextureUnit
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    float color[4]={0.f,0.f,0.f,0.f};
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_3D,GL_TEXTURE_BORDER_COLOR,color);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage3D(GL_TEXTURE_3D,0,GL_RED,volume_x,volume_y,volume_z,0,GL_RED,GL_UNSIGNED_BYTE,volume->GetData());
    GL_CHECK
}

void MultiVolumeRender::ResetVolumeSpace(float x, float y, float z) noexcept {
    if(space_x==x && space_y==y && space_z==z)
        return;

    this->space_x=x;
    this->space_y=y;
    this->space_z=z;
    setVolumeBoard();
    //change space will also change visible board
    setVisibleBoard();
}

void MultiVolumeRender::SetVisibleX(float x0, float x1) noexcept {
    if(x0==this->x0 && x1==this->x1)
        return;
    this->x0=x0;
    this->x1=x1;
    setVisibleBoard();
}
void MultiVolumeRender::SetVisibleY(float y0, float y1) noexcept {
    if(this->y0==y0 && this->y1==y1)
        return;
    this->y0=y0;
    this->y1=y1;
    setVisibleBoard();
}
void MultiVolumeRender::SetVisibleZ(float z0, float z1) noexcept {
    if(this->z0==z0 && this->z1==z1)
        return;
    this->z0=z0;
    this->z1=z1;
    setVisibleBoard();
}

void MultiVolumeRender::SetSlicer(std::shared_ptr<Slicer> slicer) noexcept {
    this->slicer=slicer;
    this->slice_visible=true;
    setSlice();
}

void MultiVolumeRender::resize(int w, int h) noexcept{
    if(w>0 && h>0 && w<MAX_SLICE_W && h<MAX_SLICE_H){
        HWND window=WindowFromDC(this->window_handle);
        MoveWindow(window,0,0,w,h,false);
        this->window_width=w;
        this->window_height=h;
    }
}

void MultiVolumeRender::initGL() {
    auto ins=GetModuleHandle(NULL);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::string idx=std::to_string(dist(rng));
    HWND window=create_window(ins,("wgl_invisable"+idx).c_str(),window_width,window_height);
    this->window_handle=GetDC(window);
    this->gl_context=create_opengl_context(this->window_handle);
    glEnable(GL_DEPTH_TEST);
    spdlog::info("successfully init OpenGL context.");
}

void MultiVolumeRender::SetVisible(bool volume, bool slice) noexcept {
    this->volume_visible=volume;
    this->slice_visible=slice;

}

void MultiVolumeRender::bindShaderUniform() {
    multi_volume_render_shader->use();
    multi_volume_render_shader->setInt("transfer_func",0);
    multi_volume_render_shader->setInt("preInt_transferfunc",1);
    multi_volume_render_shader->setInt("volume_data",2);

    multi_volume_render_shader->setFloat("ka",0.5f);
    multi_volume_render_shader->setFloat("kd",0.8f);
    multi_volume_render_shader->setFloat("shininess",100.0f);
    multi_volume_render_shader->setFloat("ks",1.0f);
    multi_volume_render_shader->setVec3("light_direction",glm::normalize(glm::vec3(-1.0f,-1.0f,-1.0f)));

    multi_volume_render_shader->setFloat("space_x",space_x);
    multi_volume_render_shader->setFloat("space_y",space_y);
    multi_volume_render_shader->setFloat("space_z",space_z);

    multi_volume_render_shader->setVec3("volume_board",volume_board_x,volume_board_y,volume_board_z);

    multi_volume_render_shader->setBool("slice_visible",slice_visible);

    slice_render_shader->use();
    slice_render_shader->setInt("volume_data",2);
    slice_render_shader->setVec3("volume_board",volume_board_x,volume_board_y,volume_board_z);
}

void MultiVolumeRender::bindTextureUnit() {
    glBindTextureUnit(0,transfer_func_tex);
    glBindTextureUnit(1,preInt_tf_tex);
    glBindTextureUnit(2,volume_tex);
    glBindTextureUnit(3,raycast_entry_pos_tex);
    glBindTextureUnit(4,raycast_exit_pos_tex);
    glBindTextureUnit(5,slice_color_tex);
    glBindTextureUnit(6,slice_pos_tex);
    GL_CHECK
}

void MultiVolumeRender::render() noexcept {
//    spdlog::info("{0}",__FUNCTION__ );
    setSlice();
    bindTextureUnit();
    bindShaderUniform();
    GL_CHECK
    glm::mat4 view=glm::lookAt(glm::vec3{camera.pos[0],camera.pos[1],camera.pos[2]},
                               glm::vec3{camera.pos[0]+camera.front[0],camera.pos[1]+camera.front[1],camera.pos[2]+camera.front[2]},
                               glm::vec3{camera.up[0],camera.up[1],camera.up[2]});
    glm::mat4 projection=glm::perspective(glm::radians(camera.zoom),(float)window_width/window_height,camera.n,camera.f);
    glm::mat4 mvp=projection*view;
    //1. render slice and volume board position

    //2. render volume and slice
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    volume_render_pos_shader->use();
    volume_render_pos_shader->setMat4("MVPMatrix",mvp);

    glBindFramebuffer(GL_FRAMEBUFFER,raycast_pos_fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(volume_visible_board_vao);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,0);

    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glDrawBuffer(GL_COLOR_ATTACHMENT1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,0);
    glDisable(GL_CULL_FACE);

    slice_render_shader->use();
    slice_render_shader->setMat4("MVPMatrix",mvp);

//    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    static GLenum drawBuffers[ 2 ] = { GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    if(!volume_visible){
        drawBuffers[0]=GL_COLOR_ATTACHMENT0;
        glBindFramebuffer(GL_FRAMEBUFFER,0);
    }
    else{
        drawBuffers[0]=GL_COLOR_ATTACHMENT2;
        glBindFramebuffer(GL_FRAMEBUFFER,raycast_pos_fbo);

    }

//    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    if(slice_visible){
        glDrawBuffers(2,drawBuffers);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(slice_vao);
        glDrawArrays(GL_TRIANGLES,0,6);
    }

//    glBindVertexArray(volume_visible_board_vao);
//        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
//        glDisable(GL_DEPTH_TEST);
//    glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,0);
//    glEnable(GL_DEPTH_TEST);
//    glPolygonMode(GL_FRONT_AND_BACK,GL_TRIANGLES);
    if(volume_visible){
        glBindFramebuffer(GL_FRAMEBUFFER,0);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        multi_volume_render_shader->use();
        glBindVertexArray(screen_quad_vao);
        glDrawArrays(GL_TRIANGLES,0,6);
    }

//    if(!volume_visible && !slice_visible){
//        glBindFramebuffer(GL_FRAMEBUFFER,0);
//        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//    }
    glFinish();

    GL_CHECK

}

void MultiVolumeRender::RenderSlice() noexcept {

}

void MultiVolumeRender::setVolumeBoard() {
    this->volume_board_x=volume_x*space_x;
    this->volume_board_y=volume_y*space_y;
    this->volume_board_z=volume_z*space_z;
    volume_board_indices={
            0,1,2,0,2,3,
            0,4,1,4,5,1,
            1,5,6,6,2,1,
            6,7,2,7,3,2,
            7,4,3,3,4,0,
            4,7,6,4,6,5
    };
    volume_board_vertices[0]={0.0f,0.0f,0.0f};
    volume_board_vertices[1]={volume_board_x,0.0f,0.0f};
    volume_board_vertices[2]={volume_board_x,volume_board_y,0.0f};
    volume_board_vertices[3]={0.0f,volume_board_y,0.0f};
    volume_board_vertices[4]={0.0f,0.0f,volume_board_z};
    volume_board_vertices[5]={volume_board_x,0.0f,volume_board_z};
    volume_board_vertices[6]={volume_board_x,volume_board_y,volume_board_z};
    volume_board_vertices[7]={0.0f,volume_board_y,volume_board_z};

    if(volume_board_vao && volume_board_vbo && volume_board_ebo){
        glNamedBufferSubData(volume_board_vbo,0,sizeof(volume_board_vertices),volume_board_vertices.data());
    }
    else{
        glGenVertexArrays(1,&volume_board_vao);
        glGenBuffers(1,&volume_board_vbo);
        glGenBuffers(1,&volume_board_ebo);

        glBindVertexArray(volume_board_vao);
        glBindBuffer(GL_ARRAY_BUFFER,volume_board_vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(volume_board_vertices),volume_board_vertices.data(),GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,volume_board_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(volume_board_indices),volume_board_indices.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(GLfloat),(void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    GL_CHECK

}

void MultiVolumeRender::setVisibleBoard() {
    float v_x0=x0*volume_board_x;
    float v_x1=x1*volume_board_x;
    float v_y0=y0*volume_board_y;
    float v_y1=y1*volume_board_y;
    float v_z0=z0*volume_board_z;
    float v_z1=z1*volume_board_z;

    volume_visible_board_indices={
            0,1,2,0,2,3,
            0,4,1,4,5,1,
            1,5,6,6,2,1,
            6,7,2,7,3,2,
            7,4,3,3,4,0,
            4,7,6,4,6,5
    };

    volume_visible_board_vertices[0]={v_x0,v_y0,v_z0};
    volume_visible_board_vertices[1]={v_x1,v_y0,v_z0};
    volume_visible_board_vertices[2]={v_x1,v_y1,v_z0};
    volume_visible_board_vertices[3]={v_x0,v_y1,v_z0};
    volume_visible_board_vertices[4]={v_x0,v_y0,v_z1};
    volume_visible_board_vertices[5]={v_x1,v_y0,v_z1};
    volume_visible_board_vertices[6]={v_x1,v_y1,v_z1};
    volume_visible_board_vertices[7]={v_x0,v_y1,v_z1};

    if(volume_visible_board_vao && volume_visible_board_vbo && volume_visible_board_ebo){
        //just update
        glNamedBufferSubData(volume_visible_board_vbo,0,sizeof(volume_visible_board_vertices),volume_visible_board_vertices.data());
    }
    else{
        glGenVertexArrays(1,&volume_visible_board_vao);
        glGenBuffers(1,&volume_visible_board_vbo);
        glGenBuffers(1,&volume_visible_board_ebo);

        glBindVertexArray(volume_visible_board_vao);
        glBindBuffer(GL_ARRAY_BUFFER,volume_visible_board_vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(volume_visible_board_vertices),volume_visible_board_vertices.data(),GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,volume_visible_board_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(volume_visible_board_indices),volume_visible_board_indices.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(GLfloat),(void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }


    GL_CHECK

}

void MultiVolumeRender::setPosFrameBuffer() {
    glGenFramebuffers(1,&raycast_pos_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,raycast_pos_fbo);

    glGenTextures(1,&raycast_entry_pos_tex);
    glBindTexture(GL_TEXTURE_2D,raycast_entry_pos_tex);
//    glBindTextureUnit
    glTextureStorage2D(raycast_entry_pos_tex,1,GL_RGBA32F,window_width,window_height);
    glBindImageTexture(0,raycast_entry_pos_tex,0,GL_FALSE,0,GL_READ_WRITE,GL_RGBA32F);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,raycast_entry_pos_tex,0);
    GL_CHECK;

    glGenRenderbuffers(1,&raycast_pos_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER,raycast_pos_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,window_width,window_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, raycast_pos_rbo);

    glGenTextures(1,&raycast_exit_pos_tex);
    glBindTexture(GL_TEXTURE_2D,raycast_exit_pos_tex);
//    glBindTextureUnit
    glTextureStorage2D(raycast_exit_pos_tex,1,GL_RGBA32F,window_width,window_height);
    glBindImageTexture(1,raycast_exit_pos_tex,0,GL_FALSE,0,GL_READ_WRITE,GL_RGBA32F);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,raycast_exit_pos_tex,0);
    GL_CHECK


    glGenTextures(1,&slice_color_tex);
    glBindTexture(GL_TEXTURE_2D,slice_color_tex);
    glTextureStorage2D(slice_color_tex,1,GL_RGBA32F,window_width,window_height);
    glBindImageTexture(2,slice_color_tex,0,GL_FALSE,0,GL_READ_WRITE,GL_RGBA32F);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,slice_color_tex,0);
    GL_CHECK

    glGenTextures(1,&slice_pos_tex);
    glBindTexture(GL_TEXTURE_2D,slice_pos_tex);
    glTextureStorage2D(slice_pos_tex,1,GL_RGBA32F,window_width,window_height);
    glBindImageTexture(3,slice_pos_tex,0,GL_FALSE,0,GL_READ_WRITE,GL_RGBA32F);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT3,GL_TEXTURE_2D,slice_pos_tex,0);
    GL_CHECK

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){
        throw std::runtime_error("Framebuffer object is not complete!");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL_CHECK
}

void MultiVolumeRender::setScreenQuad() {
    screen_quad_vertices={
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

void MultiVolumeRender::setShader() {
    this->slice_render_shader=std::make_unique<Shader>(
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\slice_render_v.glsl",
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\slice_render_f.glsl"
            );
    this->volume_render_pos_shader=std::make_unique<Shader>(
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\volume_render_pos_v.glsl",
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\volume_render_pos_f.glsl"
            );
    this->multi_volume_render_shader=std::make_unique<Shader>(
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\multi_volume_render_v.glsl",
            "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\multi_volume_render_f.glsl"
            );
}

void MultiVolumeRender::SetTransferFunction(TransferFunc &&tf) noexcept {
    TransferFuncImpl tf_impl(tf);

    glGenTextures(1,&transfer_func_tex);
    glBindTexture(GL_TEXTURE_1D,transfer_func_tex);
//    glBindTextureUnit
    glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D,0,GL_RGBA,TF_DIM,0,GL_RGBA,GL_FLOAT,tf_impl.getTransferFunction().data());

    GL_CHECK

    glGenTextures(1,&preInt_tf_tex);
    glBindTexture(GL_TEXTURE_2D,preInt_tf_tex);
//    glBindTextureUnit
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,TF_DIM,TF_DIM,0,GL_RGBA,GL_FLOAT, tf_impl.getPreIntTransferFunc().data());

    GL_CHECK
}

auto MultiVolumeRender::GetFrame() noexcept -> Frame {
    Frame frame;
    frame.width=window_width;
    frame.height=window_height;
    frame.channels=4;
    frame.data.resize((size_t)window_height*window_width*4);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glReadPixels(0,0,window_width,window_height,GL_RGBA,GL_UNSIGNED_BYTE,reinterpret_cast<void*>(frame.data.data()));
    GL_CHECK
    return frame;
}

void MultiVolumeRender::SetCamera(Camera camera) noexcept {
    this->camera=camera;
}

void MultiVolumeRender::clear() noexcept {

}

void MultiVolumeRender::setSlice() {
    auto calcSliceV=[&]()->void{
        auto slice=slicer->GetSlice();
        glm::vec3 origin= {slice.origin[0],slice.origin[1],slice.origin[2]};
        glm::vec3 up={slice.up[0],slice.up[1],slice.up[2]};
        glm::vec3 right={slice.right[0],slice.right[1],slice.right[2]};
        auto lu=origin+up*slice.voxel_per_pixel_height*(float)slice.n_pixels_height/2.f
                -right*slice.voxel_per_pixel_width*(float)slice.n_pixels_width/2.f;
        auto lb=origin-up*slice.voxel_per_pixel_height*(float)slice.n_pixels_height/2.f
                -right*slice.voxel_per_pixel_width*(float)slice.n_pixels_width/2.f;
        auto ru=origin+up*slice.voxel_per_pixel_height*(float)slice.n_pixels_height/2.f
                +right*slice.voxel_per_pixel_width*(float)slice.n_pixels_width/2.f;
        auto rb=origin-up*slice.voxel_per_pixel_height*(float)slice.n_pixels_height/2.f
                +right*slice.voxel_per_pixel_width*(float)slice.n_pixels_width/2.f;
        glm::vec3 space={space_x,space_y,space_z};
        lu=lu * space;
        lb=lb * space;
        ru=ru * space;
        rb=rb * space;
//        spdlog::info("lu {0} {1} {2}",lu.x,lu.y,lu.z);
//        spdlog::info("lb {0} {1} {2}",lb.x,lb.y,lb.z);
//        spdlog::info("ru {0} {1} {2}",ru.x,ru.y,ru.z);
//        spdlog::info("rb {0} {1} {2}",rb.x,rb.y,rb.z);
        slice_vertices={
                lu.x,lu.y,lu.z,
                lb.x,lb.y,lb.z,
                ru.x,ru.y,ru.z,

                ru.x,ru.y,ru.z,
                lb.x,lb.y,lb.z,
                rb.x,rb.y,rb.z
        };
    };
    if(!slice_vao || !slice_vbo){
        calcSliceV();
        glGenVertexArrays(1,&slice_vao);
        glGenBuffers(1,&slice_vbo);
        glBindVertexArray(slice_vao);
        glBindBuffer(GL_ARRAY_BUFFER,slice_vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(slice_vertices),slice_vertices.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        GL_CHECK
//        spdlog::info("init slice vertices");
    }
    else{
        if(slicer->IsModified()){
            calcSliceV();
            glNamedBufferSubData(slice_vbo,0,sizeof(slice_vertices),slice_vertices.data());
            GL_CHECK
//            spdlog::info("update slice vertices");
        }
    }
}


VS_END
