//
// Created by wyz on 2021/10/26.
//
#include "SlicerServerApplication.hpp"
#include "Server/RequestHandlerFactory.hpp"
#include <Poco/Net/HTTPServer.h>
#include <VolumeSlicer/Utils/logger.hpp>
#include "Manager/ManagerClient.hpp"
#include "Common/utils.hpp"
#include "Dataset/CompVolume.hpp"
VS_START
namespace remote
{
void SlicerServerApplication::initialize(Poco::Util::Application &self)
{
}
void SlicerServerApplication::defineOptions(Poco::Util::OptionSet &options)
{
}
void SlicerServerApplication::hanldle_option(const std::string &name, const std::string &value)
{
}
int SlicerServerApplication::main(const std::vector<std::string> &args)
{
    SetCUDACtx(0);
    VolumeDataSet::Load("E:/MouseNeuronData/mouse_file_config.json");
    max_server_num = 6;
    MessageQueue::set_queue_type("slice");

    std::unique_ptr<ManagerClient> manager = nullptr;
    std::string address="127.0.0.1:9876";
    manager = std::make_unique<ManagerClient>(address);

    std::unique_ptr<Poco::Net::HTTPServer> server = nullptr;
    Poco::Net::ServerSocket svs(16689);
    server = std::make_unique<Poco::Net::HTTPServer>(Poco::makeShared<RequestHandlerFactory>(), svs,
                                                     Poco::makeAuto<Poco::Net::HTTPServerParams>());
    server->start();
    LOG_INFO("SlicerServer start at port {0}", 16689);

    waitForTerminationRequest();

    return Application::EXIT_OK;
}
}
VS_END