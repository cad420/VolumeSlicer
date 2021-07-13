//
// Created by wyz on 2021/7/12.
//

#include "MPIVolumeSliceApplication.hpp"
#include "VolumeSliceGUI.hpp"
#include <cmdline.hpp>

int MPIVolumeSliceAPP::run(int argc, char **argv) noexcept{

    cmdline::parser cmd;


    try{
        VolumeSliceGUI gui;
        gui.init("slice_config.json");
        gui.set_comp_volume("E:\\MouseNeuronData/mouse_file_config.json");
        gui.show();
    }
    catch (const std::exception& err) {
        std::cout<<err.what()<<std::endl;
    }
    return 0;
}

