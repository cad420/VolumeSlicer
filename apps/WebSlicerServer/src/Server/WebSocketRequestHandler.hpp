//
// Created by wyz on 2021/10/26.
//
#pragma once
#include <VolumeSlicer/export.hpp>
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPServerResponse.h>

VS_START
namespace remote
{
class WebSocketRequestHandler : public Poco::Net::HTTPRequestHandler
{
  public:
    WebSocketRequestHandler();

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response) override;

  private:
};
}

VS_END
