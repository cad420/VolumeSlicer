//
// Created by wyz on 2021/10/26.
//
#include "WebSocketRequestHandler.hpp"
#include <Poco/Net/NetException.h>
#include <Poco/Net/WebSocket.h>
#include <Poco/Util/Application.h>
#include <VolumeSlicer/Utils/logger.hpp>

#include "SlicerServerApplication.hpp"
#include <DataModel/Slice.hpp>
#include <VolumeSlicer/Render/volume_sampler.hpp>
#include <atomic>
#include <iostream>
#include <seria/deserialize/rapidjson.hpp>
VS_START
namespace remote
{
WebSocketRequestHandler::WebSocketRequestHandler()
{
    MessageQueue::set_queue_type("slice");
    message_queue = &MessageQueue::get_instance();
}

void WebSocketRequestHandler::handleRequest(Poco::Net::HTTPServerRequest &request,
                                            Poco::Net::HTTPServerResponse &response)
{
    using WebSocket = Poco::Net::WebSocket;
    static std::atomic<int> server_count = 0;
    try
    {
        auto buffer_size = 4 * 1024 * 1024;
        std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_size]);
        uint32_t flags = 0;
        int received;
        bool should_close;

        if(server_count>=SlicerServerApplication::GetServerCap()){
            throw std::runtime_error("Current server num is reach to max server num");
        }
        server_count++;
        WebSocket ws(request, response);
        std::cout<<"connect websocket uri: "<<request.getURI()<<std::endl;
        LOG_INFO("New ws and current serve num is: {0}",server_count);
        auto one_hour = Poco::Timespan(0, 1, 0, 0, 0);
        ws.setReceiveTimeout(one_hour);

        auto handler =[&ws](const uint8_t* data,uint32_t size){
            ws.sendFrame(data,size,WebSocket::FRAME_BINARY);
        };

        do
        {
            received = ws.receiveFrame(buffer.get(), buffer_size, reinterpret_cast<int &>(flags));

            should_close = received <= 0 || ((flags & WebSocket::FRAME_OP_BITMASK) == WebSocket::FRAME_OP_CLOSE);
            if (should_close)
            {
                break;
            }

            message_queue->add_message(buffer.get(),buffer_size,handler);

        } while (true);
    }
    catch (Poco::Net::WebSocketException &exception)
    {
        LOG_ERROR(exception.what());
        switch (exception.code())
        {
        case WebSocket::WS_ERR_HANDSHAKE_UNSUPPORTED_VERSION:
            response.set("Sec-WebSocket-Version", WebSocket::WEBSOCKET_VERSION);
        case WebSocket::WS_ERR_NO_HANDSHAKE:
        case WebSocket::WS_ERR_HANDSHAKE_NO_VERSION:
        case WebSocket::WS_ERR_HANDSHAKE_NO_KEY:
            response.setStatusAndReason(Poco::Net::HTTPServerResponse::HTTP_BAD_REQUEST);
            response.setContentLength(0);
            response.send();
            break;
        }
    }
    catch (...)
    {
        LOG_ERROR("WebSocket finished with unknown exception");
    }
    LOG_INFO("WebSocket closed...");
    LOG_INFO("Close ws and current serve num is: {0}",server_count-1);
    server_count--;
}
}
VS_END
