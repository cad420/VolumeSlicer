//
// Created by wyz on 2021/7/30.
//
#include "opengl_comp_render_impl.hpp"

#include <VolumeSlicer/Algorithm/memory_helper.hpp>
#include <VolumeSlicer/Utils/gl_helper.hpp>
#include <VolumeSlicer/Utils/timer.hpp>

#include <iostream>
#include <random>

#include "shaders.hpp"
#include "Common/transfer_function_impl.hpp"

#define TF_TEX_UNIT 0
#define PTF_TEX_UNIT 1
#define ENTRY_TEX_UNIT 2
#define EXIT_TEX_UNIT 3
#define VOLUME_TEXTURE_UNIT_0 4

VS_START

std::unique_ptr<OpenGLCompVolumeRenderer> OpenGLCompVolumeRenderer::Create(int w, int h,bool create_opengl_context)
{
    return std::make_unique<OpenGLCompVolumeRendererImpl>(w, h,create_opengl_context);
}

OpenGLCompVolumeRendererImpl::OpenGLCompVolumeRendererImpl(int w, int h,bool create_opengl_context) : window_w(w), window_h(h)
{
    // init opengl context
    if(create_opengl_context){
        if (glfwInit() == GLFW_FALSE)
        {
            std::cout << "Failed to init GLFW" << std::endl;
            return;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_DOUBLEBUFFER, true);
        window = glfwCreateWindow(window_w, window_h, "HideWindow", nullptr, nullptr);
        if (window == nullptr)
        {
            throw std::runtime_error("Create GLFW window failed.");
        }
        make_opengl_context = [this](){
            glfwMakeContextCurrent(window);
        };
        setCurrentCtx();
        glfwHideWindow(window);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            throw std::runtime_error("GLAD failed to load opengl api");
        }
       LOG_INFO("OpenGLCompVolumeRendererImpl create glfw hide window and opengl context");
    }
    else{
        LOG_INFO("OpenGLCompVolumeRendererImpl use external opengl context");
        if(glEnable == nullptr){
            throw std::runtime_error("OpenGLCompVolumeRendererImpl not find external opengl context!");
        }
        make_opengl_context = [](){return;};
    }
    GL_EXPR(glEnable(GL_DEPTH_TEST));
    GL_EXPR(glDisable(GL_BLEND));
    GL_EXPR(glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));
    GL_CHECK
    LOG_INFO("successfully init OpenGLCompVolumeRendererImpl OpenGL context.");
    // create shader
    OpenGLCompVolumeRendererImpl::resize(w, h);

    createShader();
    createScreenQuad();
    createPosFramebuffer();
}

void OpenGLCompVolumeRendererImpl::SetVolume(std::shared_ptr<CompVolume> comp_volume)
{
    this->comp_volume = comp_volume;
    createVolumeBoundary();
    createVolumeSampler();
    this->opengl_volume_block_cache = OpenGLVolumeBlockCache::Create();
    this->opengl_volume_block_cache->SetCacheBlockLength(comp_volume->GetBlockLength()[0]);
    {
        int num;
        MemoryHelper::GetRecommendGPUTextureNum<uint8_t>(num);
        LOG_INFO("OpenGLVolumeRenderer: Recommend GPU texture num({})",num);
        this->opengl_volume_block_cache->SetCacheCapacity(num, MemoryHelper::DefaultGPUTextureSizeX, MemoryHelper::DefaultGPUTextureSizeY, MemoryHelper::DefaultGPUTextureSizeZ);
    }
    this->opengl_volume_block_cache->CreateMappingTable(this->comp_volume->GetBlockDim());

    this->total_lod_block_num = opengl_volume_block_cache->GetMappingTable().size() / 4;
    createMissedBlockMapping();
    uint32_t max_lod = 0, min_lod = 0xffffffff;
    {
        auto &mapping_table = this->opengl_volume_block_cache->GetMappingTable();
        createMappingTable(mapping_table.data(), mapping_table.size());

        auto &lod_mapping_table_offset = opengl_volume_block_cache->GetLodMappingTableOffset();
        for (auto &it : lod_mapping_table_offset)
        {
            if (it.first > max_lod)
                max_lod = it.first;
            if (it.first < min_lod)
                min_lod = it.first;
        }
        max_lod--;
        std::vector<uint32_t> offset; // for one block not for uint32_t
        offset.resize(max_lod + 1, 0);
        for (auto &it : lod_mapping_table_offset)
        {
            if (it.first <= max_lod)
                offset.at(it.first) = it.second / 4;
        }
        block_offset = std::move(offset);
        comp_render_pass_shader->use();
        comp_render_pass_shader->setUIArray("lod_mapping_table_offset", block_offset.data(), block_offset.size());
    }

    createVolumeSampler();
    auto volume_tex_handles = opengl_volume_block_cache->GetOpenGLTextureHandles();
    std::vector<int> tex_binding;
    for (int i = 0; i < volume_tex_handles.size(); i++)
    {
        GL_EXPR(glBindTextureUnit(VOLUME_TEXTURE_UNIT_0 + i, volume_tex_handles[i]));
        GL_EXPR(glBindSampler(VOLUME_TEXTURE_UNIT_0 + i, gl_volume_sampler));
        tex_binding.push_back(VOLUME_TEXTURE_UNIT_0 + i);
    }
    GL_CHECK

    comp_render_pass_shader->use();

    for (int i = 0; i < volume_tex_handles.size(); i++)
    {
        comp_render_pass_shader->setIntArray("cacheVolumes", tex_binding.data(), tex_binding.size());
    }

    auto volume_tex_shape = opengl_volume_block_cache->GetCacheShape();
    comp_render_pass_shader->setVec3("volume_texture_shape", volume_tex_shape[1], volume_tex_shape[2],
                                     volume_tex_shape[3]);
    comp_render_pass_shader->setIVec3("volume_dim", comp_volume->GetVolumeDimX(), comp_volume->GetVolumeDimY(),
                                      comp_volume->GetVolumeDimZ());
    auto block_dim = comp_volume->GetBlockDim(0);
    comp_render_pass_shader->setIVec3("block_dim", block_dim[0], block_dim[1], block_dim[2]);
    auto block_length = comp_volume->GetBlockLength();
    comp_render_pass_shader->setInt("block_length", block_length[0]);
    comp_render_pass_shader->setInt("padding", block_length[1]);
    comp_render_pass_shader->setInt("no_padding_block_length", block_length[0] - 2 * block_length[1]);
    comp_render_pass_shader->setInt("max_lod", block_length[3]);
    comp_render_pass_shader->setVec3("volume_space", comp_volume->GetVolumeSpaceX(), comp_volume->GetVolumeSpaceY(),
                                     comp_volume->GetVolumeSpaceZ());
}

void OpenGLCompVolumeRendererImpl::SetCamera(Camera camera)
{
    this->camera = camera;
}

void OpenGLCompVolumeRendererImpl::SetTransferFunc(TransferFunc tf)
{
    setCurrentCtx();
    TransferFuncImpl tf_impl(tf);
    if (!transfer_func_tex)
    {
        GL_EXPR(glGenTextures(1, &transfer_func_tex));
        GL_EXPR(glBindTexture(GL_TEXTURE_1D, transfer_func_tex));
        //    glBindTextureUnit
        GL_EXPR(glBindTextureUnit(TF_TEX_UNIT, transfer_func_tex));
        GL_EXPR(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_EXPR(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_EXPR(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL_EXPR(glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, TF_DIM, 0, GL_RGBA, GL_FLOAT, tf_impl.getTransferFunction().data()));

        GL_CHECK

        GL_EXPR(glGenTextures(1, &preInt_tf_tex));
        GL_EXPR(glBindTexture(GL_TEXTURE_2D, preInt_tf_tex));
        //    glBindTextureUnit
        GL_EXPR(glBindTextureUnit(PTF_TEX_UNIT, preInt_tf_tex));
        GL_EXPR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_EXPR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_EXPR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL_EXPR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        GL_EXPR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TF_DIM, TF_DIM, 0, GL_RGBA, GL_FLOAT,
                     tf_impl.getPreIntTransferFunc().data()));

    }
    else
    {
        GL_EXPR(glTextureSubImage1D(transfer_func_tex, 0, 0, TF_DIM, GL_RGBA, GL_FLOAT, tf_impl.getTransferFunction().data()));
        GL_EXPR(glTextureSubImage2D(preInt_tf_tex, 0, 0, 0, TF_DIM, TF_DIM, GL_RGBA, GL_FLOAT,
                            tf_impl.getPreIntTransferFunc().data()));
    }
}
void OpenGLCompVolumeRendererImpl::bindShaderUniform()
{
    comp_render_pass_shader->use();
    comp_render_pass_shader->setInt("transferFunc", TF_TEX_UNIT);
    comp_render_pass_shader->setInt("preIntTransferFunc", PTF_TEX_UNIT);

    comp_render_pass_shader->setFloat("step", step);
    comp_render_pass_shader->setInt("max_view_steps", steps);
    comp_render_pass_shader->setFloat("max_view_distance", step * steps);
    comp_render_pass_shader->setVec3("camera_pos", camera.pos[0], camera.pos[1], camera.pos[2]);
    static float volume_board_x = comp_volume->GetVolumeSpaceX() * comp_volume->GetVolumeDimX();
    static float volume_board_y = comp_volume->GetVolumeSpaceY() * comp_volume->GetVolumeDimY();
    static float volume_board_z = comp_volume->GetVolumeSpaceZ() * comp_volume->GetVolumeDimZ();
    if (camera.pos[0] < 0.f || camera.pos[1] < 0.f || camera.pos[2] < 0.f || camera.pos[0] > volume_board_x ||
        camera.pos[1] > volume_board_y || camera.pos[2] > volume_board_z)
    {
        comp_render_pass_shader->setBool("inside", false);
    }
    else
    {
        comp_render_pass_shader->setBool("inside", true);
    }
}

void OpenGLCompVolumeRendererImpl::render(bool sync)
{
    LOG_DEBUG("run into internal render");
    SCOPE_TIMER("Render a frame cost time")

    setCurrentCtx();

    bindShaderUniform();

    // get mvp from camera
    glm::mat4 view = glm::lookAt(glm::vec3{camera.pos[0], camera.pos[1], camera.pos[2]},
                                 glm::vec3{camera.look_at[0], camera.look_at[1], camera.look_at[2]},
                                 glm::vec3{camera.up[0], camera.up[1], camera.up[2]});
    //todo near and far according to space
    glm::mat4 projection = glm::perspective(glm::radians(camera.zoom), (float)window_w / window_h, 0.1f, 2000.f);
    projection[0][0] *= mpi.mpi_world_col_num;
    projection[1][1] *= mpi.mpi_world_row_num;
    projection[2][0] = -mpi.mpi_world_col_num + 1 + 2 * mpi.mpi_node_x_index;
    projection[2][1] = mpi.mpi_world_row_num - 1 - 2 * (mpi.mpi_world_row_num - 1 - mpi.mpi_node_y_index);
    glm::mat4 mvp = projection * view;

    GL_EXPR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_EXPR(glClearColor(0.0f, 0.f, 0.f, 1.0f));
    GL_EXPR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        // 1. get entry and exit pos
    {
        SCOPE_TIMER("Render ray entry and exit pos")
        GL_EXPR(glBindFramebuffer(GL_FRAMEBUFFER, raycast_pos_fbo));
        GL_EXPR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        comp_render_pos_shader->use();
        comp_render_pos_shader->setMat4("MVPMatrix", mvp);

        GL_EXPR(glBindVertexArray(volume_board_vao));
        GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT0));
        GL_EXPR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        GL_EXPR(glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0));

        GL_EXPR(glEnable(GL_CULL_FACE));
        GL_EXPR(glFrontFace(GL_CCW));
        GL_EXPR(glDrawBuffer(GL_COLOR_ATTACHMENT1));
        GL_EXPR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        GL_EXPR(glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0));
        GL_EXPR(glDisable(GL_CULL_FACE));
    }

    // 2. render pass
    // 2.1 calculate missed blocks
    {
        SCOPE_TIMER("Calculate missed blocks cost time")
        GL_EXPR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        comp_render_pass_shader->use();
        GL_EXPR(glBindVertexArray(screen_quad_vao));
        comp_render_pass_shader->setBool("render", false);
        GL_EXPR(glDrawArrays(GL_TRIANGLES, 0, 6));
        GL_EXPR(glFinish());
    }

    // 2.2 load missed blocks
    {
        AutoTimer timer1("Upload volume block cost time");
        calcMissedBlocks();

        filterMissedBlocks();

        sendRequests();

        do
        {
            fetchBlocks(sync);
        } while (sync && !isRenderFinish());

        if (sync)
        {
            assert(isRenderFinish());
        }

        clearCurrentInfo();
    }

    auto &m = opengl_volume_block_cache->GetMappingTable();
    uploadMappingTable(m.data(), m.size());

    GL_EXPR(glFlush());

    // 2.2 ray cast
    {
        AutoTimer timer1("RayCast cost time");
        comp_render_pass_shader->setBool("render", true);
        GL_EXPR(glDrawArrays(GL_TRIANGLES, 0, 6));

        GL_EXPR(glFinish());
    }

    //debug window for renderdoc
    //glfwSwapBuffers(window);

    GL_CHECK
}

auto OpenGLCompVolumeRendererImpl::GetImage() -> const Image<Color4b> &
{
    setCurrentCtx();
    GL_EXPR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_EXPR(glReadPixels(0, 0, window_w, window_h, GL_RGBA, GL_UNSIGNED_BYTE, reinterpret_cast<void *>(image.GetData())));
    return image;
}

void OpenGLCompVolumeRendererImpl::resize(int w, int h)
{
    setCurrentCtx();
    image = Image<Color4b>(w, h);
    GL_EXPR(glViewport(0, 0, w, h));
}

void OpenGLCompVolumeRendererImpl::clear()
{
    setCurrentCtx();
    deleteGLResource();
    comp_volume.reset();
}

void OpenGLCompVolumeRendererImpl::SetRenderPolicy(CompRenderPolicy policy)
{
    setCurrentCtx();
    float lod_dist[10];
    for (int i = 0; i < 10; i++)
        lod_dist[i] = policy.lod_dist[i];
    comp_render_pass_shader->use();
    comp_render_pass_shader->setFloatArray("lod_dist", lod_dist, 10);
}

void OpenGLCompVolumeRendererImpl::SetMPIRender(MPIRenderParameter mpi)
{
    this->mpi = mpi;
}

void OpenGLCompVolumeRendererImpl::SetStep(double step, int steps)
{
    this->step = step;
    this->steps = steps;
}
//=========================================================================

/**
 * @brief Calculate missed blocks for each frame render in CPU
 */
void OpenGLCompVolumeRendererImpl::calcMissedBlocks()
{
    std::unordered_set<std::array<uint32_t, 4>> cur_missed_blocks;
    for (uint32_t lod = 0; lod < block_offset.size(); lod++)
    {
        auto lod_block_dim = comp_volume->GetBlockDim(lod);
        for (uint32_t idx = block_offset[lod];
             idx < (lod + 1 < block_offset.size() ? block_offset[lod + 1] : total_lod_block_num); idx++)
        {
            if (!mapping_missed_blocks[idx])
                continue;
            uint32_t index = idx - block_offset[lod];
            uint32_t z = index / lod_block_dim[0] / lod_block_dim[1];
            uint32_t y = (index - z * lod_block_dim[0] * lod_block_dim[1]) / lod_block_dim[0];
            uint32_t x = (index % (lod_block_dim[0] * lod_block_dim[1])) % lod_block_dim[0];
            cur_missed_blocks.insert({x, y, z, lod});
        }
    }
    if(!cur_missed_blocks.empty())
        memset(mapping_missed_blocks, 0, total_lod_block_num * sizeof(uint32_t));

    // sort missed blocks by distance to camera pos
    for (auto &it : cur_missed_blocks)
    {
        if (missed_blocks.find(it) == missed_blocks.end())
        {
            new_missed_blocks.insert(it);
        }
    }

    for (auto &it : missed_blocks)
    {
        if (cur_missed_blocks.find(it) == cur_missed_blocks.end())
        {
            no_missed_blocks.insert(it);
        }
    }
    this->missed_blocks = std::move(cur_missed_blocks);
}

void OpenGLCompVolumeRendererImpl::filterMissedBlocks()
{
    if (!new_missed_blocks.empty())
    {
        std::unordered_set<std::array<uint32_t, 4>> tmp;
        for (auto &it : new_missed_blocks)
        {
            bool cached = this->opengl_volume_block_cache->SetCachedBlockValid(it);
            if (cached)
            {
            }
            else
            {
                tmp.insert(it);
            }
        }
        new_missed_blocks = std::move(tmp);
    }

    if (!no_missed_blocks.empty())
    {
        for (auto &it : no_missed_blocks)
        {
            this->opengl_volume_block_cache->SetBlockInvalid(it);
        }
    }
}

void OpenGLCompVolumeRendererImpl::sendRequests()
{
    this->comp_volume->PauseLoadBlock();
    {
        if (!missed_blocks.empty())
        {
            std::vector<std::array<uint32_t, 4>> targets;
            targets.reserve(missed_blocks.size());
            for (auto &it : missed_blocks)
                targets.push_back(it);
            comp_volume->ClearBlockInQueue(targets);
        }

        std::vector<std::array<uint32_t, 4>> tmp;
        tmp.insert(tmp.end(), new_missed_blocks.begin(), new_missed_blocks.end());
        static float volume_space_x = comp_volume->GetVolumeSpaceX();
        static float volume_space_y = comp_volume->GetVolumeSpaceY();
        static float volume_space_z = comp_volume->GetVolumeSpaceZ();
        float no_padding_block_length = comp_volume->GetBlockLength()[0] - 2 * comp_volume->GetBlockLength()[1];
        auto camera_pos = camera.pos;
        camera_pos[0] = camera_pos[0] / volume_space_x;
        camera_pos[1] = camera_pos[1] / volume_space_y;
        camera_pos[2] = camera_pos[2] / volume_space_z;
        std::sort(tmp.begin(), tmp.end(),
                  [camera_pos, no_padding_block_length](std::array<uint32_t, 4> const &v1,
                                                        std::array<uint32_t, 4> const &v2) {
                      if (v1[3] == v2[3])
                      {
                          float lod_t = std::pow(2, v1[3]) * no_padding_block_length;
                          float d1 =
                              ((v1[0] + 0.5f) * lod_t - camera_pos[0]) * ((v1[0] + 0.5f) * lod_t - camera_pos[0]) +
                              ((v1[1] + 0.5f) * lod_t - camera_pos[1]) * ((v1[1] + 0.5f) * lod_t - camera_pos[1]) +
                              ((v1[2] + 0.5f) * lod_t - camera_pos[2]) * ((v1[2] + 0.5f) * lod_t - camera_pos[2]);
                          float d2 =
                              ((v2[0] + 0.5f) * lod_t - camera_pos[0]) * ((v2[0] + 0.5f) * lod_t - camera_pos[0]) +
                              ((v2[1] + 0.5f) * lod_t - camera_pos[1]) * ((v2[1] + 0.5f) * lod_t - camera_pos[1]) +
                              ((v2[2] + 0.5f) * lod_t - camera_pos[2]) * ((v2[2] + 0.5f) * lod_t - camera_pos[2]);
                          return d1 < d2;
                      }
                      else
                      {
                          return v1[3] < v2[3];
                      }
                  });
        for (auto &it : tmp)
        {
            comp_volume->SetRequestBlock(it);
        }

        for (auto &it : no_missed_blocks)
        {
            comp_volume->EraseBlockInRequest(it);
        }
    }
    this->comp_volume->StartLoadBlock();
}

void OpenGLCompVolumeRendererImpl::clearCurrentInfo()
{
    new_missed_blocks.clear();
    no_missed_blocks.clear();
}

void OpenGLCompVolumeRendererImpl::fetchBlocks(bool sync)
{
    this->is_render_finish = true;
    if (sync)
    {
        auto tmp = new_missed_blocks;
        for (auto &it : tmp)
        {
            auto block = comp_volume->GetBlock(it);
            if (block.valid)
            {
                assert(block.block_data->GetDataPtr());
                this->opengl_volume_block_cache->UploadVolumeBlock(block.index, block.block_data->GetDataPtr(),
                                                                   block.block_data->GetSize(), true);
                block.Release();
                new_missed_blocks.erase(it);
            }
            else
            {
                this->is_render_finish = false;
            }
        }
    }
    else
    {
        for (auto &it : missed_blocks)
        {
            auto block = comp_volume->GetBlock(it);
            if (block.valid)
            {
                assert(block.block_data->GetDataPtr());
                this->opengl_volume_block_cache->UploadVolumeBlock(block.index, block.block_data->GetDataPtr(),
                                                                   block.block_data->GetSize(), true);
                block.Release();
            }
            else
            {
                this->is_render_finish = false;
            }
        }
    }
}

void OpenGLCompVolumeRendererImpl::uploadMappingTable(const uint32_t *data, size_t size)
{
    GL_EXPR(glNamedBufferSubData(mapping_table_ssbo, 0, size * sizeof(uint32_t), data));
}

void OpenGLCompVolumeRendererImpl::createMappingTable(const uint32_t *data, size_t size)
{
    GL_EXPR(glGenBuffers(1, &mapping_table_ssbo));
    GL_EXPR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, mapping_table_ssbo));
    GL_EXPR(glBufferData(GL_SHADER_STORAGE_BUFFER, size * sizeof(uint32_t), data, GL_DYNAMIC_DRAW));
    // binding point is 0
    GL_EXPR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mapping_table_ssbo));
}

void OpenGLCompVolumeRendererImpl::createMissedBlockMapping()
{
    GL_EXPR(glGenBuffers(1, &mapping_missed_blocks_ssbo));
    GL_EXPR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, mapping_missed_blocks_ssbo));
    GL_EXPR(glBufferStorage(GL_SHADER_STORAGE_BUFFER, total_lod_block_num * sizeof(uint32_t), nullptr,
                            GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT));
    this->mapping_missed_blocks = (uint32_t *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);

    GL_EXPR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mapping_missed_blocks_ssbo));
}

void OpenGLCompVolumeRendererImpl::createVolumeBoundary()
{
    setCurrentCtx();
    float volume_board_x = comp_volume->GetVolumeDimX() * comp_volume->GetVolumeSpaceX();
    float volume_board_y = comp_volume->GetVolumeDimY() * comp_volume->GetVolumeSpaceY();
    float volume_board_z = comp_volume->GetVolumeDimZ() * comp_volume->GetVolumeSpaceZ();
    std::array<std::array<float, 3>, 8> volume_board_vertices = {
        std::array<float, 3>{0.f, 0.f, 0.f},
        std::array<float, 3>{volume_board_x, 0.f, 0.f},
        std::array<float, 3>{volume_board_x, volume_board_y, 0.f},
        std::array<float, 3>{0.f, volume_board_y, 0.f},
        std::array<float, 3>{0.f, 0.f, volume_board_z},
        std::array<float, 3>{volume_board_x, 0.f, volume_board_z},
        std::array<float, 3>{volume_board_x, volume_board_y, volume_board_z},
        std::array<float, 3>{0.f, volume_board_y, volume_board_z}};
    std::array<uint32_t, 36> volume_board_indices = {0, 1, 2, 0, 2, 3, 0, 4, 1, 4, 5, 1, 1, 5, 6, 6, 2, 1,
                                                     6, 7, 2, 7, 3, 2, 7, 4, 3, 3, 4, 0, 4, 7, 6, 4, 6, 5};
    GL_EXPR(glGenVertexArrays(1, &volume_board_vao));
    GL_EXPR(glGenBuffers(1, &volume_board_vbo));
    GL_EXPR(glGenBuffers(1, &volume_board_ebo));

    GL_EXPR(glBindVertexArray(volume_board_vao));
    GL_EXPR(glBindBuffer(GL_ARRAY_BUFFER, volume_board_vbo));
    GL_EXPR(glBufferData(GL_ARRAY_BUFFER, sizeof(volume_board_vertices), volume_board_vertices.data(), GL_STATIC_DRAW));
    GL_EXPR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, volume_board_ebo));
    GL_EXPR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(volume_board_indices), volume_board_indices.data(), GL_STATIC_DRAW));
    GL_EXPR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void *)0));
    GL_EXPR(glEnableVertexAttribArray(0));
    GL_EXPR(glBindVertexArray(0));
}

void OpenGLCompVolumeRendererImpl::createScreenQuad()
{
    setCurrentCtx();
    std::array<float, 24> screen_quad_vertices = {
        -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

    GL_EXPR(glGenVertexArrays(1, &screen_quad_vao));
    GL_EXPR(glGenBuffers(1, &screen_quad_vbo));
    GL_EXPR(glBindVertexArray(screen_quad_vao));
    GL_EXPR(glBindBuffer(GL_ARRAY_BUFFER, screen_quad_vbo));
    GL_EXPR(glBufferData(GL_ARRAY_BUFFER, sizeof(screen_quad_vertices), screen_quad_vertices.data(), GL_STATIC_DRAW));
    GL_EXPR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));
    GL_EXPR(glEnableVertexAttribArray(0));
    GL_EXPR(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float))));
    GL_EXPR(glEnableVertexAttribArray(1));
    GL_EXPR(glBindVertexArray(0));

}

void OpenGLCompVolumeRendererImpl::createShader()
{
    this->comp_render_pos_shader = std::make_unique<Shader>();
    this->comp_render_pos_shader->setShader(shader::comp_render_pos_v, shader::comp_render_pos_f);
    this->comp_render_pass_shader = std::make_unique<Shader>();
    this->comp_render_pass_shader->setShader(shader::comp_render_v, shader::comp_render_f);
}
void OpenGLCompVolumeRendererImpl::createPosFramebuffer()
{
    setCurrentCtx();

    GL_EXPR(glGenFramebuffers(1, &raycast_pos_fbo));
    GL_EXPR(glBindFramebuffer(GL_FRAMEBUFFER, raycast_pos_fbo));

    GL_EXPR(glGenTextures(1, &raycast_entry_pos_tex));
    GL_EXPR(glBindTexture(GL_TEXTURE_2D, raycast_entry_pos_tex));
    //    glBindTextureUnit
    GL_EXPR(glBindTextureUnit(ENTRY_TEX_UNIT, raycast_entry_pos_tex));
    GL_EXPR(glTextureStorage2D(raycast_entry_pos_tex, 1, GL_RGBA32F, window_w, window_h));
    GL_EXPR(glBindImageTexture(0, raycast_entry_pos_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F));
    GL_EXPR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, raycast_entry_pos_tex, 0));

    GL_EXPR(glGenRenderbuffers(1, &raycast_pos_rbo));
    GL_EXPR(glBindRenderbuffer(GL_RENDERBUFFER, raycast_pos_rbo));
    GL_EXPR(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, window_w, window_h));
    GL_EXPR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, raycast_pos_rbo));

    GL_EXPR(glGenTextures(1, &raycast_exit_pos_tex));
    GL_EXPR(glBindTexture(GL_TEXTURE_2D, raycast_exit_pos_tex));
    //    glBindTextureUnit
    GL_EXPR(glBindTextureUnit(EXIT_TEX_UNIT, raycast_exit_pos_tex));
    GL_EXPR(glTextureStorage2D(raycast_exit_pos_tex, 1, GL_RGBA32F, window_w, window_h));
    GL_EXPR(glBindImageTexture(1, raycast_exit_pos_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F));
    GL_EXPR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, raycast_exit_pos_tex, 0));

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        throw std::runtime_error("Framebuffer object is not complete!");
    }
    GL_EXPR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}
void OpenGLCompVolumeRendererImpl::createVolumeSampler()
{
    GL_EXPR(glCreateSamplers(1, &gl_volume_sampler));
    GL_EXPR(glSamplerParameterf(gl_volume_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_EXPR(glSamplerParameterf(gl_volume_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    float color[4] = {0.f, 0.f, 0.f, 0.f};
    GL_EXPR(glSamplerParameterf(gl_volume_sampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER));
    GL_EXPR(glSamplerParameterf(gl_volume_sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
    GL_EXPR(glSamplerParameterf(gl_volume_sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));

    GL_EXPR(glSamplerParameterfv(gl_volume_sampler, GL_TEXTURE_BORDER_COLOR, color));
}

OpenGLCompVolumeRendererImpl::~OpenGLCompVolumeRendererImpl()
{
    deleteGLResource();

    glfwDestroyWindow(window);
}

auto OpenGLCompVolumeRendererImpl::GetBackendName() -> std::string
{
    return "opengl";
}

bool OpenGLCompVolumeRendererImpl::isRenderFinish()
{
    return is_render_finish;
}

void OpenGLCompVolumeRendererImpl::deleteGLResource()
{
    setCurrentCtx();
    opengl_volume_block_cache.reset();
    GL_EXPR(glDeleteVertexArrays(1, &volume_board_vao));
    GL_EXPR(glDeleteBuffers(1, &volume_board_vbo));
    GL_EXPR(glDeleteBuffers(1, &volume_board_ebo));
    GL_EXPR(glDeleteVertexArrays(1, &screen_quad_vao));
    GL_EXPR(glDeleteBuffers(1, &screen_quad_vbo));
    GL_EXPR(glDeleteBuffers(1, &screen_quad_ebo));
    GL_EXPR(glDeleteFramebuffers(1, &raycast_pos_fbo));
    GL_EXPR(glDeleteRenderbuffers(1, &raycast_pos_rbo));
    GL_EXPR(glDeleteTextures(1, &raycast_entry_pos_tex));
    GL_EXPR(glDeleteTextures(1, &raycast_exit_pos_tex));
    GL_EXPR(glDeleteBuffers(1, &mapping_table_ssbo));
    GL_EXPR(glUnmapNamedBuffer(mapping_missed_blocks_ssbo));
    GL_EXPR(glDeleteBuffers(1, &mapping_missed_blocks_ssbo));
    GL_EXPR(glDeleteTextures(1, &transfer_func_tex));
    GL_EXPR(glDeleteTextures(1, &preInt_tf_tex));
    GL_EXPR(glDeleteSamplers(1, &gl_volume_sampler));
}

VS_END

#undef TF_TEX_UNIT
#undef PTF_TEX_UNIT
#undef ENTRY_TEX_UNIT
#undef EXIT_TEX_UNIT
#undef VOLUME_TEXTURE_UNIT_0