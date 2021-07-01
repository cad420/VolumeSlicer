//
// Created by wyz on 2021/6/8.
//

#ifndef VOLUMESLICER_RENDER_HPP
#define VOLUMESLICER_RENDER_HPP

#include<VolumeSlicer/volume.hpp>

VS_START

class Camera;
class TransferFunc;
class Frame;

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

    virtual auto GetFrame() noexcept ->Frame =0;

    virtual void resize(int w,int h) noexcept =0;

    //clear volume and slicer
    virtual void clear() noexcept=0;
};


using RawVolumeRenderer=Renderer<RawVolume>;

std::unique_ptr<RawVolumeRenderer> CreateRenderer(int w,int h);

VS_END

#endif //VOLUMESLICER_RENDER_HPP
