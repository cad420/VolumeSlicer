//
// Created by wyz on 2021/6/8.
//

#pragma once

#include <VolumeSlicer/Common/frame.hpp>
#include <VolumeSlicer/Data/mesh.hpp>
#include <VolumeSlicer/Data/volume.hpp>
#include <VolumeSlicer/Render/camera.hpp>
#include <VolumeSlicer/Render/transfer_function.hpp>

VS_START

template<class T,class enable= void>
class Renderer;

/**
 * @brief No window renderer for slice and raw-volume mix render
 * This renderer will no more update and may exits some bugs now.
 */
template<class T>
class VS_EXPORT Renderer<T,typename std::enable_if<T::value>::type>{
public:

    Renderer()=default;

    virtual ~Renderer()=default;

    /**
     * @brief Set camera for next render.
     * @sa Camera
     */
    virtual void SetCamera(Camera camera) noexcept =0;

    /**
     * @brief Set transfer function for next render.
     * @sa TransferFunc
     */
    virtual void SetTransferFunction(TransferFunc&& tf) noexcept =0;

    /**
     * @brief Set interpolated transfer function data(RGBA32F).
     * @param dim number of RGBA components
     */
    virtual void SetTransferFunc1D(float* data,int dim=256) noexcept = 0;

    virtual void SetVolume(const std::shared_ptr<T>& volume) noexcept =0;

    //x0 and x1 are within 0.f~1.f, and x0<=x1
    /**
     * @brief Set volume visible board range.
     * @param x0 x-dim volume visible start range.
     * @param x1 x-dim volume visible stop range. (0.0 <= x0 <= x1 <= 1.0)
     */
    virtual void SetVisibleX(float x0,float x1) noexcept =0;

    virtual void SetVisibleY(float y0,float y1) noexcept =0;

    virtual void SetVisibleZ(float z0,float z1) noexcept =0;

    /**
     * @param slicer shared_ptr for Slicer
     */
    virtual void SetSlicer(std::shared_ptr<Slicer> slicer) noexcept =0;

    /**
     * @param volume if draw volume
     * @param slice if draw slice
     */
    virtual void SetVisible(bool volume,bool slice) noexcept =0;

    /**
     * @brief Render a frame and store it.
     * @sa GetFrame
     */
    virtual void render() noexcept =0;

    /**
     * @brief Get last rendered frame.
     * @sa Frame
     */
    virtual auto GetFrame() noexcept -> Frame =0;

    virtual void resize(int w,int h) noexcept =0;

    /**
     * @brief Clear slicer and volume set
     */
    virtual void clear() noexcept=0;
};

using SliceRawVolumeMixRenderer =Renderer<RawVolume>;

[[deprecated]] VS_EXPORT std::unique_ptr<SliceRawVolumeMixRenderer> CreateRenderer(int w,int h);



/**
 * @brief Struct for mpi render parameters definition
 */
struct alignas(16) MPIRenderParameter{
    float mpi_node_x_offset = 0.f;//node center to world center in index
    float mpi_node_y_offset = 0.f;
    int mpi_world_window_w = 0;
    int mpi_world_window_h = 0;
    int mpi_world_col_num = 1;
    int mpi_world_row_num = 1;
    int mpi_node_x_index = 0;
    int mpi_node_y_index = 0;
};

/**
 * @brief Interface for volume render
 */
class VS_EXPORT IVolumeRenderer{
public:
    IVolumeRenderer() = default;

    virtual ~IVolumeRenderer(){}

    /**
     * @return renderer backend name, one of {"opengl","vulkan","cuda","cpu"}
     */
    virtual auto GetBackendName()-> std::string = 0;

    virtual void SetMPIRender(MPIRenderParameter) = 0;

    /**
     * @param step raycast step, may set to 0 meanings set by volume space itself
     * @param steps raycast steps, may set to 0 meanings raycast from entry pos to exit pos,
     * this param is mainly used for comp-volume render
     */
    virtual void SetStep(double step,int steps) = 0;

    virtual void SetCamera(Camera camera) = 0;

    virtual void SetTransferFunc(TransferFunc tf) = 0;

    /**
     * @brief render a frame according to current status (MPIRenderParameter, step and steps, Camera, TransferFunc)
     * @param sync if sync is true then render will wait all resources loaded and then render a frame,
     * if sync if false then render will start immediately and no wait for others
     */
    virtual void render(bool sync) = 0;

    /**
     * @brief this should call after render otherwise would get old result rendered by last call of render
     */
    virtual auto GetImage()->const Image<Color4b>& = 0;

    virtual void resize(int w,int h) = 0;

    /**
     * @brief clear all resources
     */
    virtual void clear() = 0;
};

class VS_EXPORT IRawVolumeRenderer: public IVolumeRenderer{
public:
    virtual void SetVolume(std::shared_ptr<RawVolume> raw_volume) = 0;

};

class VS_EXPORT CUDARawVolumeRenderer: public IRawVolumeRenderer{
public:
    static std::unique_ptr<CUDARawVolumeRenderer> Create(int w,int h,CUcontext ctx = nullptr);
};

class VS_EXPORT OpenGLRawVolumeRenderer: public IRawVolumeRenderer{
public:
    static std::unique_ptr<OpenGLRawVolumeRenderer> Create(int w,int h);
};

class VS_EXPORT VulkanRawVolumeRenderer: public IRawVolumeRenderer{
    static std::unique_ptr<VulkanRawVolumeRenderer> Create(int w,int h);
};

class VS_EXPORT CPURawVolumeRenderer: public IRawVolumeRenderer{
public:
    static std::unique_ptr<CPURawVolumeRenderer> Create(int w,int h);
};

/**
 * @brief Lod policy for large-block-volume render.
 * @relatesalso Level of Details
 */
struct CompRenderPolicy{
    double lod_dist[10];// lod i for distance x that lod[i-1] < x <= lod[i]
    [[deprecated]] std::string volume_value_file;
    [[deprecated]] std::string avg_value_file;
    [[deprecated]] std::string cdf_value_file;//not suit for comp-volume render
};

/**
 * @brief Interface for large-block-volume(comp-volume) render
 */
class VS_EXPORT ICompVolumeRenderer: public IVolumeRenderer{
public:
    virtual void SetVolume(std::shared_ptr<CompVolume> comp_volume) = 0;

    virtual void SetRenderPolicy(CompRenderPolicy) = 0;

};

class VS_EXPORT CUDACompVolumeRenderer: public ICompVolumeRenderer{
public:
    /**
     * @param ctx if nullptr is pass for Create, implement will call GetCUDACtx() to get global cuda context,
     * which means the renderer itself will not create new cuda context.
     */
    static std::unique_ptr<CUDACompVolumeRenderer> Create(int w,int h,CUcontext ctx=nullptr);
};

class VS_EXPORT OpenGLCompVolumeRenderer: public ICompVolumeRenderer{
public:
    /**
     * @param create_opengl_context if false the internal renderer will try to direct using opengl function without
     * create own opengl context than meanings it share the opengl context in the caller context.
     */
    static std::unique_ptr<OpenGLCompVolumeRenderer> Create(int w,int h,bool create_opengl_context = true);
};

/**
 * @brief this not imply in this project now, but had implied in another project.
 * Using vulkan renderer's efficiency is about same vs opengl,
 * but vulkan can handle with a host compute with multi-GPUs.
 */
class VS_EXPORT VulkanCompVolumeRenderer: public ICompVolumeRenderer{
    static std::unique_ptr<VulkanCompVolumeRenderer> Create(int w,int h);
};

class VS_EXPORT IOffScreenCompVolumeRenderer: public ICompVolumeRenderer{
  public:

};

/**
 * @note Although cpu renderer, it still need set CompVolume which means it need CUDA support.
 */
class VS_EXPORT CPUOffScreenCompVolumeRenderer: public IOffScreenCompVolumeRenderer{
  public:
    static std::unique_ptr<CPUOffScreenCompVolumeRenderer> Create(int w,int h);
};

class VS_EXPORT CUDAOffScreenCompVolumeRenderer: public IOffScreenCompVolumeRenderer{
  public:
    static std::unique_ptr<CUDAOffScreenCompVolumeRenderer> Create(int w,int h,CUcontext ctx=nullptr);
};

/**
 * @brief Interface for mesh renderer, some are the same with IVolumeRenderer.
 * @sa IVolumeRenderer IMeshLoaderPluginInterface
 */
class VS_EXPORT IMeshRenderer{
  public:
    virtual ~IMeshRenderer(){}

    virtual void SetMesh(std::shared_ptr<Mesh> mesh) = 0;

    virtual void SetCamera(Camera camera) = 0;

    virtual void SetMPIRender(MPIRenderParameter mpi) = 0;

    virtual void render() = 0;

    virtual auto GetImage()-> const Image<Color4b>& = 0;

    virtual void resize(int w,int h) = 0;

    virtual void clear() = 0;
};

/**
 * @brief Renderer for simple neuron display, just very simple.
 * @sa IMeshLoaderPluginInterface
 */
class VS_EXPORT SimpleMeshRenderer: public IMeshRenderer{
  public:
    static std::unique_ptr<SimpleMeshRenderer> Create(int w,int h);
};

VS_END


