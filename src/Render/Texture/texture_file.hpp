//
// Created by wyz on 2021/8/26.
//
#pragma once

#include "texture.hpp"
VS_START

class TextureFile{
  public:
    template<typename Texel>
    static Texture1D<Texel> LoadTexture1DFromMemory(const Texel* tex_data,uint32_t length){
        using CoordType=typename TextureBase<Texel,1>::CoordType;
        TextureBase<Texel,1> base(CoordType{length});
        if(base.IsAvailable()){
            memcpy(base.RawData(),tex_data,sizeof(Texel)*length);
        }
        else
            throw std::bad_alloc();
        return Texture1D<Texel>(std::move(base));
    }
    template <typename Texel>
    static Texture2D<Texel> LoadTexture2DFromMemory(const Texel* tex_data,uint32_t width,uint32_t height){
        using CoordType=typename TextureBase<Texel,2>::CoordType;
        TextureBase<Texel,2> base(CoordType{width,height});
        if(base.IsAvailable()){
            memcpy(base.RawData(),tex_data,sizeof(Texel)*width*height);
        }
        else
            throw std::bad_alloc();
        return Texture2D<Texel>(std::move(base));
    }
    template <typename Texel>
    static Texture3D<Texel> LoadTexture3DFromMemory(const Texel* tex_data,uint32_t x,uint32_t y,uint32_t z){
        using CoordType=typename TextureBase<Texel,3>::CoordType;
        TextureBase<Texel,3> base(CoordType{x,y,z});
        if(base.IsAvailable()){
            memcpy(base.RawData(),tex_data,sizeof(Texel)*x*y*z);
        }
        else
            throw std::bad_alloc();
        return Texture3D<Texel>(std::move(base));
    }

    template <typename Texel>
    static Texture3D<Texel> LoadTexture3DFromCUDAMemory(const Texel* tex_data,uint32_t x,uint32_t y,uint32_t z);

};

VS_END


#ifdef CUDA_IMPL

#include <VolumeSlicer/cuda_context.hpp>
VS_START

template <typename Texel>
Texture3D<Texel> TextureFile::LoadTexture3DFromCUDAMemory(const Texel *tex_data, uint32_t x, uint32_t y, uint32_t z)
{
    using CoordType=typename TextureBase<Texel,3>::CoordType;
    TextureBase<Texel,3> base(CoordType{x,y,z});
    if(base.IsAvailable()){
        cudaMemcpy(base.RawData(),tex_data,sizeof(Texel)*x*y*z,cudaMemcpyDefault);
    }
    else
        throw std::bad_alloc();
    return Texture3D<Texel>(std::move(base));
}

VS_END

#endif