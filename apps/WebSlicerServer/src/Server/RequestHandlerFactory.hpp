//
// Created by wyz on 2021/10/26.
//
#pragma once
#include <VolumeSlicer/export.hpp>
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPServerRequest.h>

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