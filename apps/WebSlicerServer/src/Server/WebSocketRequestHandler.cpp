//
// Created by wyz on 2021/10/26.
//
#include "WebSocketRequestHandler.hpp"
#include <Poco/Net/NetException.h>
#include <Poco/Net/WebSocket.h>
#include <Poco/Util/Application.h>
#include <Utils/logger.hpp>

#include <DataModel/Slice.hpp>
#include <seria/deserialize.hpp>
#include <iostream>
#include <VolumeSlicer/volume_sampler.hpp>
VS_START
namespace remote
{
WebSocketRequestHandler::WebSocketRequestHandler()
{
}

void WebSocketRequestHandler::handleRequest(Poco::Net::HTTPServerRequest &request,
                                            Poco::Net::HTTPServerResponse &response)
{
    using WebSocket = Poco::Net::WebSocket;

    try
    {
        auto buffer_size = 4 * 1024 * 1024;
        std::unique_ptr<char[]> buffer(new char[buffer_size]);
        uint32_t flags = 0;
        int received;
        bool should_close;

        WebSocket ws(request, response);
        auto one_hour = Poco::Timespan(0, 1, 0, 0, 0);
        ws.setReceiveTimeout(one_hour);

        rapidjson::Document doc;
        SetCUDACtx(0);
        auto comp_volume = CompVolume::Load("E:/MouseNeuronData/mouse_file_config.json");
        comp_volume->SetSpaceX(0.00032);
        comp_volume->SetSpaceY(0.00032);
        comp_volume->SetSpaceZ(0.001);
        auto slice_sampler = VolumeSampler::CreateVolumeSampler(std::move(comp_volume));
        do
        {
            received = ws.receiveFrame(buffer.get(), buffer_size, reinterpret_cast<int &>(flags));

            should_close = received <= 0 || ((flags & WebSocket::FRAME_OP_BITMASK) == WebSocket::FRAME_OP_CLOSE);
            if (should_close)
            {
                break;
            }

            try{
                doc.Parse(buffer.get(),received);
                if(doc.HasParseError() || !doc.IsObject()){
                    throw std::runtime_error("RapidJson parse error");
                }
                auto objects = doc.GetObject();
                if(doc.HasMember("slice")){
                    auto values = objects["slice"].GetObject();
                    Slice slice;
                    static std::vector<uint8_t> image;
                    seria::deserialize(slice,values);
                    image.resize(slice.n_pixels_height*slice.n_pixels_width);
                    slice_sampler->Sample(slice,image.data(),false);
                    ws.sendFrame(image.data(),image.size(),WebSocket::FRAME_BINARY);
                }
            }
            catch (const std::exception& err)
            {
                LOG_ERROR("Serialize to json cause exception: {0}.",err.what());
                ws.sendFrame(err.what(),std::strlen(err.what()),WebSocket::FRAME_TEXT);
            }

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
}
}
VS_END
