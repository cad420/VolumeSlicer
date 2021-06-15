//
// Created by wyz on 2021/6/8.
//

#ifndef VOLUMESLICER_RENDER_HPP
#define VOLUMESLICER_RENDER_HPP

#include<VolumeSlicer/volume.hpp>

VS_START

class Camera;
class TransferFunc;

template<class T,class enable= void>
class Renderer;

/**
 * no window renderer, provide frame after every call render with camera and tf config
 */
template<class T>
class VS_EXPORT Renderer<T,typename std::enable_if<T::value>::type>{
public:
    Renderer()=default;
    virtual ~Renderer()=default;
    virtual void SetCamera(Camera&& camera) noexcept =0;
    virtual void SetTransferFunction(TransferFunc&& tf) noexcept =0;
    virtual void SetVolume(std::shared_ptr<T> volume) noexcept =0;
    virtual void SetSlice(const Slice&) noexcept =0;
    virtual void SetSlice(Slice&&) noexcept =0;
    virtual void render() noexcept =0;
    virtual void render(const Slice&) noexcept =0;
    virtual void GetFrame() noexcept =0;
    virtual void resize(int w,int h) noexcept =0;
    virtual void clear() noexcept=0;
};

using RawVolumeRenderer=Renderer<Volume<VolumeType::Raw>>;

VS_END

#endif //VOLUMESLICER_RENDER_HPP
