//
// Created by wyz on 2021/6/8.
//

#ifndef VOLUMESLICER_RENDER_HPP
#define VOLUMESLICER_RENDER_HPP

#include<VolumeSlicer/volume.hpp>
#include<VolumeSlicer/camera.hpp>
#include<VolumeSlicer/frame.hpp>
#include<VolumeSlicer/transfer_function.hpp>
VS_START


template<class T,class enable= void>
class Renderer;

/**
 * no window renderer, provide frame after every call render with camera and tf config
 *
 */
template<class T>
class VS_EXPORT Renderer<T,typename std::enable_if<T::value>::type>{
public:
    Renderer()=default;
    virtual ~Renderer()=default;

    //just receive class Camera, camera's operation processing should be imply other place in client.
    //if not set, renderer will try to use last saved camera
    //camera's pos should according to volume's space and dim
    virtual void SetCamera(Camera camera) noexcept =0;

    virtual void SetTransferFunction(TransferFunc&& tf) noexcept =0;

    virtual void SetTransferFunc1D(float* data,int dim=256) noexcept = 0;

    //in general, volume data would not modify since loaded.
    //this function just load volume data upto GPU's texture once.
    virtual void SetVolume(const std::shared_ptr<T>& volume) noexcept =0;

    //reset space for current volume data. todo should delete this interface
    virtual void ResetVolumeSpace(float x,float y,float z) noexcept =0;

    //x0 and x1 are within 0.f~1.f, and x0<=x1
    virtual void SetVisibleX(float x0,float x1) noexcept =0;
    virtual void SetVisibleY(float y0,float y1) noexcept =0;
    virtual void SetVisibleZ(float z0,float z1) noexcept =0;

    //notice arg is Slicer not Slice
    //a Slicer contain a Slice
    //Slice may be modified frequently so pass a shared_ptr of Slicer
    virtual void SetSlicer(std::shared_ptr<Slicer> slicer) noexcept =0;

    //set if volume or slice should draw
    virtual void SetVisible(bool volume,bool slice) noexcept =0;

    //just volume render: volume and slice mix render
    virtual void render() noexcept =0;

    //!render the slice and save result to member slicer which had set
    //!this result is not for GetFrame, it equal to sample but use OpenGL not CUDA
    virtual void RenderSlice() noexcept =0;

    virtual auto GetFrame() noexcept -> Frame =0;

    virtual void resize(int w,int h) noexcept =0;

    //clear volume and slicer
    virtual void clear() noexcept=0;
};


using RawVolumeRenderer=Renderer<RawVolume>;

VS_EXPORT std::unique_ptr<RawVolumeRenderer> CreateRenderer(int w,int h);

struct MPIRenderParameter{
    float mpi_node_x_offset;
    float mpi_node_y_offset;
    int mpi_world_window_w;
    int mpi_world_window_h;
};

class VS_EXPORT IVolumeRenderer{
public:
    virtual ~IVolumeRenderer(){}

    virtual void SetMPIRender(MPIRenderParameter) = 0;

    virtual void SetStep(double step,int steps) = 0;

    virtual void SetCamera(Camera camera) = 0;

    virtual void SetTransferFunc(TransferFunc tf) = 0;

    virtual void render() = 0;

    virtual auto GetFrame()-> const Image<uint32_t>& = 0;

    virtual void resize(int w,int h) = 0;

    virtual void clear() = 0;
};
class VS_EXPORT IRawVolumeRenderer: public IVolumeRenderer{
public:
    virtual void SetVolume(std::shared_ptr<RawVolume> raw_volume) = 0;

    void SetMPIRender(MPIRenderParameter) override = 0;

    void SetStep(double step,int steps) override = 0;

    void SetCamera(Camera camera) override = 0;

    void SetTransferFunc(TransferFunc tf) override = 0;

    void render() override = 0;

    auto GetFrame()  -> const Image<uint32_t>&  override = 0;

    void resize(int w,int h) override = 0;

    void clear() override = 0;
};
class VS_EXPORT CUDARawVolumeRenderer: public IRawVolumeRenderer{
public:
    static std::unique_ptr<CUDARawVolumeRenderer> Create(int w,int h,CUcontext ctx=nullptr);
};
class VS_EXPORT OpenGLRawVolumeRenderer: public IRawVolumeRenderer{
public:
    static std::unique_ptr<OpenGLRawVolumeRenderer> Create(int w,int h);
};
class VS_EXPORT CPURawVolumeRenderer: public IRawVolumeRenderer{
public:
    static std::unique_ptr<CPURawVolumeRenderer> Create(int w,int h);
    virtual auto GetImage()->const Image<Color4b>& = 0;
    struct RenderParameter{
        double step;

    };
};

struct CompRenderPolicy{
    double lod_dist[10];// lod i for distance x that lod[i-1] < x <= lod[i]
    std::string volume_value_file;
    std::string avg_value_file;
    std::string cdf_value_file;
};

class VS_EXPORT ICompVolumeRenderer: public IVolumeRenderer{
public:
    virtual void SetVolume(std::shared_ptr<CompVolume> comp_volume) = 0;

    virtual void SetRenderPolicy(CompRenderPolicy) = 0;

    void SetMPIRender(MPIRenderParameter) override = 0;

    void SetStep(double step,int steps) override = 0;

    void SetCamera(Camera camera) override = 0;

    void SetTransferFunc(TransferFunc tf) override = 0;

    void render() override = 0;

    auto GetFrame()  -> const Image<uint32_t>&  override = 0;

    void resize(int w,int h) override = 0;

    void clear() override = 0;
};

class VS_EXPORT CUDACompVolumeRenderer: public ICompVolumeRenderer{
public:
    static std::unique_ptr<CUDACompVolumeRenderer> Create(int w,int h,CUcontext ctx=nullptr);
};

class VS_EXPORT OpenGLCompVolumeRenderer: public ICompVolumeRenderer{
public:
    static std::unique_ptr<OpenGLCompVolumeRenderer> Create(int w,int h);
};

class VS_EXPORT IOffScreenCompVolumeRenderer: public ICompVolumeRenderer{
  public:
    virtual auto GetImage()->const Image<Color4b>& = 0;
};
/**
 * suitable for off-screen render
 */
class VS_EXPORT CPUOffScreenCompVolumeRenderer: public IOffScreenCompVolumeRenderer{
  public:
    static std::unique_ptr<CPUOffScreenCompVolumeRenderer> Create(int w,int h);
};

class VS_EXPORT CUDAOffScreenCompVolumeRenderer: public IOffScreenCompVolumeRenderer{
  public:
    static std::unique_ptr<CUDAOffScreenCompVolumeRenderer> Create(int w,int h,CUcontext ctx=nullptr);
};

VS_END

#endif //VOLUMESLICER_RENDER_HPP
