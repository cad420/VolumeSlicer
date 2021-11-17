//
// Created by wyz on 2021/9/13.
//
#include "OffScreenVolumeRenderer.hpp"
#include <iostream>
int main(int argc,char** argv){
    if(argc<2){
        std::cout<<"Not provide the config file path, Usage: OffScreenVolumeRenderer.exe config_file_path"<<std::endl;
        return 0;
    }
    try{
        std::cout<<"Json config file: "<<argv[1]<<std::endl;
        OffScreenVolumeRenderer::RenderFrames(argv[1]);
    }
    catch (const std::exception& err)
    {
        std::cout<<err.what()<<std::endl;
    }
    catch (...)
    {
        std::cout<<"Unknown exception happened"<<std::endl;
    }

    return 0;
}