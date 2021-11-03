//
// Created by wyz on 2021/10/29.
//
#include "ManagerClient.hpp"
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPMessage.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Utils/logger.hpp>
#include <iostream>
VS_START
namespace remote{

ManagerClient::ManagerClient(std::string address)
:address(std::move(address)),uri("http://"+this->address)
{
    MessageQueue::set_queue_type("slice");
    message_queue = &MessageQueue::get_instance();
    work = std::thread([this](){
        register_worker();
    });
}

ManagerClient::~ManagerClient()
{
    shutdown();
}

auto ManagerClient::get_address() const
{
    return address;
}

void ManagerClient::shutdown()
{
    if(ws){
        ws->shutdown();
    }
    if(work.joinable()){
        work.join();
    }
}

void ManagerClient::register_worker()
{
    using namespace Poco::Net;
    HTTPClientSession session(uri.getHost(),uri.getPort());
    HTTPRequest request(HTTPRequest::HTTP_GET,"/worker/slice/1",HTTPMessage::HTTP_1_1);
    HTTPResponse response;

    try{
        auto buffer_size = 4 * 1024 * 1024;
        std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_size]);
        uint32_t flags = 0;
        int received;
        bool should_close;

        ws = std::make_unique<WebSocket>(session,request,response);
        if(!ws){
            throw std::runtime_error("create websocket failed");
        }
        std::cout<<"connect websocket uri: "<<request.getURI()<<std::endl;
        auto one_hour = Poco::Timespan(0, 1, 0, 0, 0);
        ws->setReceiveTimeout(one_hour);

        auto handler = [ pws = ws.get()](const uint8_t* response,uint32_t total){
            pws->sendFrame(response,total,WebSocket::FRAME_BINARY);
        };

        do{
            received = ws->receiveFrame(buffer.get(), buffer_size, reinterpret_cast<int &>(flags));

            auto is_ping = (flags & WebSocket::FRAME_OP_BITMASK) == WebSocket::FRAME_OP_PING;
            if (is_ping) {
                LOG_INFO("received ping");
                ws->sendFrame(buffer.get(), received,WebSocket::FRAME_FLAG_FIN | WebSocket::FRAME_OP_PONG);
                continue;
            }

            should_close = received <= 0 || ((flags & WebSocket::FRAME_OP_BITMASK) == WebSocket::FRAME_OP_CLOSE);
            if (should_close)
            {
                break;
            }


            message_queue->add_message(buffer.get(),received,handler);

        }while(true);
        LOG_INFO("ManagerClient connection closed");
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("ManagerClient error: {0}",err.what());
    }
    catch (...)
    {
        LOG_ERROR("ManagerClient closed with unknown error");
    }
}

}
VS_END