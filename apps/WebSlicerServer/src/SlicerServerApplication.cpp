//
// Created by wyz on 2021/10/26.
//
#include "SlicerServerApplication.hpp"
#include "Server/RequestHandlerFactory.hpp"
#include <Poco/Net/HTTPServer.h>
#include <Utils/logger.hpp>
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