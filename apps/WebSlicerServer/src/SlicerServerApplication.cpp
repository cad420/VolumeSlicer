//
// Created by wyz on 2021/10/26.
//
#include "SlicerServerApplication.hpp"
#include "Server/RequestHandlerFactory.hpp"
#include <Poco/Net/HTTPServer.h>
#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/IntValidator.h>
#include <Poco/NumberParser.h>
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
    using Option = Poco::Util::Option;
    using OptionCallback = Poco::Util::OptionCallback<SlicerServerApplication>;

    ServerApplication::defineOptions(options);

    options.addOption(Option("port", "p", "port listening")
                          .required(false)
                          .argument("port")
                          .repeatable(false)
                          .validator(new Poco::Util::IntValidator(1, 65536))
                          .callback(OptionCallback(
                              this, &SlicerServerApplication::hanldle_option)));

    options.addOption(Option("manager", "m", "manager address")
                          .required(false)
                          .argument("manager")
                          .repeatable(false)
                          .callback(OptionCallback(
                              this, &SlicerServerApplication::hanldle_option)));

    options.addOption(Option("comp", "c", "comp storage path")
                          .required(false)
                          .argument("comp")
                          .repeatable(false)
                          .callback(OptionCallback(
                              this, &SlicerServerApplication::hanldle_option)));

    options.addOption(Option("raw", "r", "raw storage path")
                          .required(false)
                          .argument("raw")
                          .repeatable(false)
                          .callback(OptionCallback(
                              this, &SlicerServerApplication::hanldle_option)));

}
void SlicerServerApplication::hanldle_option(const std::string &name, const std::string &value)
{
    if (name == "port") {
        m_port = Poco::NumberParser::parse(value);
        return;
    }

    if (name == "manager") {
        m_manager_address = value;
        return;
    }

    if (name == "comp") {
        m_storage_comp = value;
        return;
    }

    if (name == "raw") {
        m_storage_raw = value;
        return;
    }
}
int SlicerServerApplication::main(const std::vector<std::string> &args)
{
#ifdef _WIN32
    for(int i=0;i<args.size();i++){
        auto pos=args[i].find('=');
        hanldle_option(args[i].substr(2,pos-2),args[i].substr(pos+1));
    }
#endif
    SetCUDACtx(0);
    if(!m_storage_comp.empty())
        VolumeDataSet::Load(m_storage_comp);
    if(!m_storage_raw.empty())
        VolumeDataSet::Load(m_storage_raw);

    max_server_num = 6;
    MessageQueue::set_queue_type("slice");

    std::unique_ptr<ManagerClient> manager = nullptr;
    std::string address=m_manager_address;
    if(!address.empty())
        manager = std::make_unique<ManagerClient>(address);

    std::unique_ptr<Poco::Net::HTTPServer> server = nullptr;
    if(m_port)
    {
        Poco::Net::ServerSocket svs(m_port);
        server = std::make_unique<Poco::Net::HTTPServer>(Poco::makeShared<RequestHandlerFactory>(), svs,
                                                         Poco::makeAuto<Poco::Net::HTTPServerParams>());
        server->start();
        LOG_INFO("SlicerServer start at port {0}", m_port);
    }

    waitForTerminationRequest();

    return Application::EXIT_OK;
}
}
VS_END