//
// Created by wyz on 2021/10/26.
//
#pragma once
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <VolumeSlicer/Common/export.hpp>

VS_START
namespace remote
{
class RequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory
{
  public:
    Poco::Net::HTTPRequestHandler *createRequestHandler(const Poco::Net::HTTPServerRequest &request) override;
};
}

VS_END