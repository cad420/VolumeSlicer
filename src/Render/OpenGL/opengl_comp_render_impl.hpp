//
// Created by wyz on 2021/7/30.
//

#pragma once

#include <VolumeSlicer/Data/cdf.hpp>
#include <VolumeSlicer/Data/volume_cache.hpp>
#include <VolumeSlicer/Render/render.hpp>
#include <VolumeSlicer/Utils/hash.hpp>
#include <unordered_set>
//#include "Render/wgl_wrap.hpp"
#include "shader_program.hpp"
//#include <glad/glad.h>
#include <GLFW/glfw3.h>

VS_START


class OpenGLCompVolumeRendererImpl:public OpenGLCompVolumeRenderer{
public:
    OpenGLCompVolumeRendererImpl(int w,int h,bool create_opengl_context);

    ~OpenGLCompVolumeRendererImpl() override;

    void SetVolume(std::shared_ptr<CompVolume> comp_volume) override;

    void SetRenderPolicy(CompRenderPolicy) override;

    auto GetBackendName()-> std::string override;

    void SetMPIRender(MPIRenderParameter) override ;

    void SetStep(double step,int steps) override;

    void SetCamera(Camera camera) override ;

    void SetTransferFunc(TransferFunc tf) override ;

    void render(bool sync) override ;

    auto GetImage()->const Image<Color4b>&  override ;

    void resize(int w,int h) override ;

    void clear() override ;

private:
    void calcMissedBlocks();

    void filterMissedBlocks();

    void sendRequests();

    void fetchBlocks(bool sync);
  private:
    void uploadMappingTable(const uint32_t* data,size_t size);

    void createMappingTable(const uint32_t* data,size_t size);

    void createMissedBlockMapping();

    void createVolumeBoundary();

    void createScreenQuad();

    void createShader();

    void createPosFramebuffer();

    void bindShaderUniform();

    void createVolumeSampler();

    bool isRenderFinish();

    void clearCurrentInfo();

    void deleteGLResource();
private:
    void setCurrentCtx(){
        make_opengl_context();
    }
private:
    std::function<void()> make_opengl_context;

    Image<Color4b> image;

    GLFWwindow* window = nullptr;
    int window_w,window_h;
    Camera camera;

    bool is_render_finish;

    std::shared_ptr<CompVolume> comp_volume;
    std::unique_ptr<OpenGLVolumeBlockCache> opengl_volume_block_cache;

    uint32_t volume_board_vao = 0,volume_board_vbo = 0,volume_board_ebo = 0;
    uint32_t screen_quad_vao = 0,screen_quad_vbo = 0,screen_quad_ebo = 0;
    uint32_t raycast_pos_fbo = 0,raycast_pos_rbo = 0,raycast_entry_pos_tex = 0,raycast_exit_pos_tex = 0;

    uint32_t total_lod_block_num = 0;
    uint32_t mapping_table_ssbo = 0;
    std::vector<uint32_t> block_offset;
    std::unordered_set<std::array<uint32_t,4>> missed_blocks;
    std::unordered_set<std::array<uint32_t,4>> new_missed_blocks,no_missed_blocks;
    uint32_t* mapping_missed_blocks = nullptr;
    uint32_t mapping_missed_blocks_ssbo = 0;

    float step = 0.f;
    int steps = 0;

    uint32_t transfer_func_tex = 0,preInt_tf_tex = 0;
    uint32_t gl_volume_sampler = 0;

    std::unique_ptr<Shader> comp_render_pos_shader;
    std::unique_ptr<Shader> comp_render_pass_shader;

    MPIRenderParameter mpi;
};


VS_END


