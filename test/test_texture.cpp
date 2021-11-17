//
// Created by wyz on 2021/8/26.
//
#include <Render/Texture/sampler.hpp>
#include <Render/Texture/texture_file.hpp>
#include <VolumeSlicer/color.hpp>
#include <iostream>
#include <vector>

#include <VolumeSlicer/Utils/block_array.hpp>
#include <VolumeSlicer/Utils/linear_array.hpp>
#include <VolumeSlicer/Utils/block_cache.hpp>
#include <cuda_runtime.h>
using namespace vs;
void test_texture();
void test_block_array();
void test_linear_array();
void test_block_cache();
int main(){
//    test_texture();
//    test_block_array();
//    test_linear_array();
    test_block_cache();
    return 0;
}
void test_block_cache(){

}
void test_linear_array(){
    struct alignas(8) A{
        char a;
        int b;
    };
    std::cout<<alignof(A)<<" "<<sizeof(A)<<std::endl;
    std::cout<<typeid(std::aligned_storage<111,8>::type).name()
              <<" "<<sizeof(std::aligned_storage<111,alignof(A)>::type)<<std::endl;
    std::cout<<typeid(sizeof(int)).name()<<std::endl;
    class B{
        int* b;
        int a;
      public:
        B(){

        }
        B(int a):a(a){

        }
        B(const B& b){}
    };
    std::cout<<"B: "<<std::is_class_v<uint8_t><<std::endl;
    std::cout<<std::is_default_constructible_v<B><<std::endl;
    std::cout<<std::is_trivially_copy_constructible_v<B><<std::endl;
    Linear3DArray<Color4b> arr_3d_color4b(5,5,5);
    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
            for(int z=0;z<5;z++){
                arr_3d_color4b(i,j,z)={i,j,z,0};
            }
        }
    }

}
void test_block_array(){
    std::cout<<"test Block3DArray......"<<std::endl;


    Block3DArray<uint8_t,3> block_3d_array(64,64,64,2);
    std::cout<<block_3d_array.BlockNumX()<<" "<<block_3d_array.BlockNumY()<<" "<<block_3d_array.BlockNumZ()<<std::endl;
    Linear3DArray<uint8_t> linear3DArray(64,64,64,3);

    Block3DArray<uint8_t,3> block_3d_array2(64,64,64,linear3DArray.RawData());
    auto ptr=block_3d_array2.GetBlockData(0,0,0);
    block_3d_array.SetBlockData(1,1,1,ptr);
    ptr=block_3d_array.GetBlockData(1,1,1);
    for(int i=0;i<8*8*8;i++)
        std::cout<<i<<" "<<(int)ptr[i]<<std::endl;
    std::cout<<"sample ret: "<<(int)block_3d_array.Sample(1,1,1,0.5,0.5,0.5)<<std::endl;
    std::cout<<"sample ret: "<<(int)block_3d_array.Sample(1,1,0,0.5,0.5,0.5)<<std::endl;
}
void test_texture(){
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
    std::vector<Color4b> tex_3d_color4b_sub(8);
    for(int i=0;i<8;i++){
        tex_3d_color4b_sub[i]={1,1,1,0};
    }
    tex_3d_color4b.UpdateTextureSubData(2,2,2,2,2,2,tex_3d_color4b_sub.data());
    std::cout<<"after update 3d texture sub data"<<std::endl;
    for(int i=2;i<4;i++){
        for(int j=2;j<4;j++){
            for(int k=2;k<4;k++){
//                auto res=LinearSampler::Sample3D(tex_3d_color4b,0.12*i,0.15*j,0.25*k);
                auto res=tex_3d_color4b.At(i,j,k);
                std::cout<<i<<" "<<j<<" "<<k<<" "<<(int)res.r<<" "<<(int)res.g<<" "<<(int)res.b<<" "<<(int)res.a<<std::endl;
            }
        }
    }

}