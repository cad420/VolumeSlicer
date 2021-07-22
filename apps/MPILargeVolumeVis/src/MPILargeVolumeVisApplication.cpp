//
// Created by wyz on 2021/7/22.
//
#include "MPILargeVolumeVisApplication.hpp"
#include "LargeVolumeVisGUI.hpp"
#include <cmdline.hpp>
int MPILargeVolumeVisApp::run(int argc, char **argv) {

    cmdline::parser cmd;

    try{
        LargeVolumeVisGUI gui;
        gui.init("slicer_config.json");
        gui.show();
    }
    catch (const std::exception& err) {
        std::cout<<err.what()<<std::endl;
    }
    return 0;
}
