//
// Created by wyz on 2021/11/5.
//
#include "MPILargeMeshVisApplication.hpp"
#include "LargeMeshVisGUI.hpp"
#include <Utils/logger.hpp>
#include <cmdline.hpp>
int MPILargeMeshVisApp::run(int argc, char **argv)
{
    try
    {
        LargeMeshVisGUI gui;
        gui.init("mesh_config.json");
        gui.show();
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("MPILargeMeshVisApp::run error {0}",err.what());
    }
    catch (...)
    {
        LOG_ERROR("MPILargeMeshVisAPP::run finished with unknown error");
    }
    return 0;
}
