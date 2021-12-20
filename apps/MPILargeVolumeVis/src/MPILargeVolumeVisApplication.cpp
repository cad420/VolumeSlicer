//
// Created by wyz on 2021/7/22.
//
#include "MPILargeVolumeVisApplication.hpp"
#include "LargeVolumeVisGUI.hpp"
#include <cmdline.hpp>
int MPILargeVolumeVisApp::run(int argc, char **argv) {

    cmdline::parser cmd;
    cmd.add<std::string>("config",'c',"config file path",true);
    cmd.parse_check(argc,argv);
    auto config = cmd.get<std::string>("config");
    try{
        LargeVolumeVisGUI gui;
        gui.init(config.c_str());
        gui.show();
    }
    catch (const std::exception& err) {
        LOG_ERROR("MPILargeVolumeVisApp run exception: {0}",err.what());
    }
    return 0;
}
