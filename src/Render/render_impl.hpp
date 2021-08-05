//
// Created by wyz on 2021/6/8.
//

#ifndef VOLUMESLICER_RENDER_IMPL_HPP
#define VOLUMESLICER_RENDER_IMPL_HPP
#include<VolumeSlicer/render.hpp>
#include<VolumeSlicer/camera.hpp>
#include<VolumeSlicer/transfer_function.hpp>
#include<VolumeSlicer/frame.hpp>

#include"Render/shader_program.hpp"

#include<Windows.h>
VS_START




/**
 * this class just support simple raw volume and surface render.
 * only use OpenGL, no use for CUDA
 */
class MultiVolumeRender: public RawVolumeRenderer {
public:
    MultiVolumeRender(int w,int h);

    ~MultiVolumeRender() override;

    void SetCamera(Camera camera) noexcept override;

    void SetTransferFunction(TransferFunc&& tf) noexcept override;

    void SetTransferFunc1D(float* tf,int dim=256) noexcept override;

    void SetVolume(const std::shared_ptr<RawVolume>& volume) noexcept override;

    void ResetVolumeSpace(float x,float y,float z) noexcept override;

    void SetVisibleX(float x0,float x1) noexcept override;

    void SetVisibleY(float y0,float y1) noexcept override;

    void SetVisibleZ(float z0,float z1) noexcept override;

    void SetSlicer(std::shared_ptr<Slicer> slicer) noexcept override;

    void SetVisible(bool volume,bool slice) noexcept override;

    void render() noexcept override;

    void RenderSlice() noexcept override;

    auto GetFrame() noexcept ->Frame override;

    void resize(int w,int h) noexcept override;

    void clear() noexcept override;
private:
    void initGL();
    void setVolumeBoard();
    void setVisibleBoard();
    void setPosFrameBuffer();
    void setScreenQuad();
    void setShader();
    void bindShaderUniform();
    void bindTextureUnit();
    void setSlice();
private:
    //wgl window
    HDC window_handle;
    HGLRC gl_context;
    uint32_t window_width,window_height;

    //volume
    GLuint volume_tex;
    uint32_t volume_x,volume_y,volume_z;
    float space_x,space_y,space_z;
    bool volume_visible;

    //volume cube board
    float volume_board_x,volume_board_y,volume_board_z;
    GLuint volume_board_vao,volume_board_vbo,volume_board_ebo;
    std::array<std::array<GLfloat,3>,8> volume_board_vertices;
    std::array<GLuint,36> volume_board_indices;
    GLuint volume_board_line_vao,volume_board_line_vbo,volume_board_line_ebo;
    std::array<GLuint,24> volume_board_line_indices;

    //volume render cube board
    float x0,x1,y0,y1,z0,z1;
    GLuint volume_visible_board_vao,volume_visible_board_vbo,volume_visible_board_ebo;
    std::array<std::array<GLfloat,3>,8> volume_visible_board_vertices;
    std::array<GLuint,36> volume_visible_board_indices;

    //slice
    std::shared_ptr<Slicer> slicer;
    bool slice_visible;
    GLuint slice_vao,slice_vbo;
    std::array<GLfloat,18> slice_vertices;

    //transfer function
    GLuint transfer_func_tex;
    GLuint preInt_tf_tex;

    //quad screen
    GLuint screen_quad_vao,screen_quad_vbo;
    std::array<GLfloat,24> screen_quad_vertices;//could not store in class member

    //raycast framebuffer
    GLuint raycast_pos_fbo,raycast_pos_rbo;
    GLuint raycast_entry_pos_tex,raycast_exit_pos_tex;
    GLuint slice_color_tex,slice_pos_tex;

    //glsl shader
    std::unique_ptr<Shader> slice_render_shader;//no need to use unique_ptr
    std::unique_ptr<Shader> volume_render_pos_shader;
    std::unique_ptr<Shader> multi_volume_render_shader;
    std::unique_ptr<Shader> volume_board_render_shader;

    //simple camera
    //may inside volume
    Camera camera;//no need to use unique_ptr
};




VS_END




#endif //VOLUMESLICER_RENDER_IMPL_HPP
