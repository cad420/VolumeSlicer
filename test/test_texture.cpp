//
// Created by wyz on 2021/8/26.
//
#include <Render/Texture/sampler.hpp>
#include <Render/Texture/texture_file.hpp>
#include <VolumeSlicer/color.hpp>
#include <iostream>
#include <vector>
using namespace vs;

int main(){
    std::vector<float> d(256,0.f);
    for(auto i=0;i<256;i++){
        d[i]=1.f*i;
    }
    auto tf=TextureFile::LoadTexture1DFromMemory(d.data(),d.size());
    for(auto i=0;i<tf.GetLength();i++){
        std::cout<<i<<" "<<tf.At(i)<<std::endl;
    }
    for(int i=0;i<100;i++){
        std::cout<<i<<" "<<LinearSampler::Sample1D(tf,0.01*i)<<std::endl;
    }

    std::vector<uint32_t> tex_2d_v(1024,0.f);
    for(int i=0;i<1024;i++)
        tex_2d_v[i]=i;
    auto tex_2d=TextureFile::LoadTexture2DFromMemory(tex_2d_v.data(),32,32);
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            std::cout<<i<<" "<<j<<" "<<LinearSampler::Sample2D(tex_2d,0.1*i,0.2*j)<<std::endl;
        }
    }
    std::vector<double> tex_3d_v(4096,0.f);
    for(int i=0;i<4096;i++){
        tex_3d_v[i]=i*1.0;
    }
    auto tex_3d=TextureFile::LoadTexture3DFromMemory(tex_3d_v.data(),16,16,16);
    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
            for(int k=0;k<5;k++){
                std::cout<<i<<" "<<j<<" "<<k<<" "<<LinearSampler::Sample3D(tex_3d,0.12*i,0.13*j,0.15*k)<<std::endl;
            }
        }
    }

    std::vector<Color4b> tex_3d_color4b_v(64);
    for(int i=0;i<64;i++){
        tex_3d_color4b_v[i]=Color4b{i,i*2,i*3,255};
    }
    auto tex_3d_color4b=TextureFile::LoadTexture3DFromMemory(tex_3d_color4b_v.data(),4,4,4);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                auto res=LinearSampler::Sample3D(tex_3d_color4b,0.12*i,0.15*j,0.25*k);
                std::cout<<i<<" "<<j<<" "<<k<<" "<<(int)res.r<<" "<<(int)res.g<<" "<<(int)res.b<<" "<<(int)res.a<<std::endl;
            }
        }
    }
    glm::vec<4,double> v{10,20,30,255};
    v= v * 1.2;
    return 0;

}