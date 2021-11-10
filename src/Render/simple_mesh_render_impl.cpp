//
// Created by wyz on 2021/11/5.
//
#include "simple_mesh_render_impl.hpp"
#include <Utils/gl_helper.hpp>
#include <Utils/logger.hpp>
#include <Utils/timer.hpp>
#include <glm/gtc/matrix_transform.hpp>
VS_START

std::unique_ptr<SimpleMeshRenderer> SimpleMeshRenderer::Create(int w, int h)
{
    return std::make_unique<SimpleMeshRendererImpl>(w,h);
}

//==========================================================

SimpleMeshRendererImpl::SimpleMeshRendererImpl(int w, int h)
:window_w(w),window_h(h)
{
    initGL();
    SimpleMeshRendererImpl::resize(w,h);
    this->simple_mesh_render_shader=std::make_unique<Shader>(
        "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\simple_mesh_render_v.glsl",
        "C:\\Users\\wyz\\projects\\VolumeSlicer\\src\\Render\\shader\\simple_mesh_render_f.glsl"
        );
}

SimpleMeshRendererImpl::~SimpleMeshRendererImpl()
{

}

void SimpleMeshRendererImpl::SetMesh(std::shared_ptr<Mesh> mesh)
{
    setCurrentCtx();

    auto& surfaces = mesh->GetAllSurfaces();
    this->surfaces_vao.resize(surfaces.size());
    this->surfaces_vbo.resize(surfaces.size());
    this->surfaces_ebo.resize(surfaces.size());
    this->surfaces_indices_num.resize(surfaces.size());
    this->color_map.resize(surfaces.size());
    glGenVertexArrays(surfaces.size(),surfaces_vao.data());
    glGenBuffers(surfaces.size(),surfaces_vbo.data());
    glGenBuffers(surfaces.size(),surfaces_ebo.data());
    for(int i=0;i<surfaces.size();i++){
        assert(surfaces[i].has_normal);
        color_map[i]=surfaces[i].color;
        surfaces_indices_num[i]=surfaces[i].indices.size();
        glBindVertexArray(surfaces_vao[i]);
        glBindBuffer(GL_ARRAY_BUFFER,surfaces_vbo[i]);
        glBufferData(GL_ARRAY_BUFFER,surfaces[i].vertices.size()*sizeof(Mesh::Vertex),
                     surfaces[i].vertices.data(),GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,surfaces_ebo[i]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,surfaces[i].indices.size()*sizeof(uint32_t),
                     surfaces[i].indices.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        GL_CHECK
    }
    GL_CHECK
    setupMeshColorMap();
}

void SimpleMeshRendererImpl::SetCamera(Camera camera)
{
    this->camera = camera;
}

void SimpleMeshRendererImpl::SetMPIRender(MPIRenderParameter mpi)
{
    this->mpi = mpi;
}

void SimpleMeshRendererImpl::render()
{
//    AutoTimer timer;
    setCurrentCtx();
    glm::mat4 view=glm::lookAt(glm::vec3{camera.pos[0],camera.pos[1],camera.pos[2]},
                               glm::vec3{camera.look_at[0],camera.look_at[1],camera.look_at[2]},
                               glm::vec3{camera.up[0],camera.up[1],camera.up[2]});
    glm::mat4 projection=glm::perspective(glm::radians(camera.zoom),(float)window_w/window_h,0.001f,3.f);
//    glm::mat4 projection = glm::ortho(-glm::radians(camera.zoom),glm::radians(camera.zoom),-glm::radians(camera.zoom),glm::radians(camera.zoom),0.001f,5.f);
    projection[0][0] *= mpi.mpi_world_col_num;
    projection[1][1] *= mpi.mpi_world_row_num;
    projection[2][0] = -mpi.mpi_world_col_num + 1 + 2* mpi.mpi_node_x_index;
    projection[2][1] = mpi.mpi_world_row_num - 1 - 2* mpi.mpi_node_y_index;

    glm::mat4 mvp=projection*view;

    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glClearColor(0.0f,0.f,0.f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    simple_mesh_render_shader->use();
    simple_mesh_render_shader->setMat4("MVPMatrix",mvp);
    simple_mesh_render_shader->setVec3("camera_pos",camera.pos[0],camera.pos[1],camera.pos[2]);
    simple_mesh_render_shader->setVec3("light_pos",camera.pos[0],camera.pos[1],camera.pos[2]);
    for(int i=0;i<surfaces_vao.size();i++){
        simple_mesh_render_shader->setInt("surface_idx",i);
        glBindVertexArray(surfaces_vao[i]);
        glDrawElements(GL_TRIANGLES,surfaces_indices_num[i],GL_UNSIGNED_INT,0);
        GL_CHECK
    }
    glFlush();
//    glFinish();
    GL_CHECK
//    glfwSwapBuffers(window);
}

auto SimpleMeshRendererImpl::GetImage() -> const Image<Color4b> &
{
    setCurrentCtx();
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glReadPixels(0,0,window_w,window_h,GL_RGBA,GL_UNSIGNED_BYTE,image.GetData());
    GL_CHECK
    return image;
}

void SimpleMeshRendererImpl::resize(int w, int h)
{
    setCurrentCtx();
    this->window_w=w;
    this->window_h=h;
    glViewport(0,0,w,h);
    image = Image<Color4b>(w,h);
}

void SimpleMeshRendererImpl::clear()
{
    setCurrentCtx();
    LOG_ERROR("Not supported yet");
}

void SimpleMeshRendererImpl::initGL()
{
    if (glfwInit() == GLFW_FALSE)
    {
        std::cout << "Failed to init GLFW" << std::endl;
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, true);
    glfwWindowHint(GLFW_SAMPLES,4);
    window=glfwCreateWindow(window_w,window_h,"HideWindow",nullptr, nullptr);
    if(window==nullptr){
        throw std::runtime_error("Create GLFW window failed.");
    }
    setCurrentCtx();
    glfwHideWindow(window);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        throw std::runtime_error("GLAD failed to load opengl api");
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
}
void SimpleMeshRendererImpl::setupMeshColorMap()
{
    glGenBuffers(1,&color_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,color_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,color_map.size()*sizeof(color_map[0]),color_map.data(),GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,color_map_ssbo);
    GL_CHECK
}

VS_END
