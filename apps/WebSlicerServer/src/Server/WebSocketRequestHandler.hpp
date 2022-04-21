//
// Created by wyz on 2021/10/26.
//
#pragma once
#include "RPC/MessageQueue.hpp"
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/Net/HTTPServerResponse.h>
#include <VolumeSlicer/Common/export.hpp>
VS_START
namespace remote
{
class WebSocketRequestHandler : public Poco::Net::HTTPRequestHandler
{
  public:
    WebSocketRequestHandler();

    WebSocketRequestHandler(const std::string& type);

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response) override;

  private:
    MessageQueue* message_queue;
};
}

VS_END
