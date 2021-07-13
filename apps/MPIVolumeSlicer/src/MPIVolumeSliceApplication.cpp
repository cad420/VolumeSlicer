//
// Created by wyz on 2021/7/12.
//

#include "MPIVolumeSliceApplication.hpp"
#include "VolumeSliceGUI.hpp"
#include <cmdline.hpp>

int MPIVolumeSliceAPP::run(int argc, char **argv) noexcept{

    cmdline::parser cmd;



    VolumeSliceGUI gui;
    gui.set_comp_volume("E:\\MouseNeuronData/mouse_file_config.json");
    gui.show();

    return 0;
}

