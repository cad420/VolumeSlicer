//
// Created by wyz on 2021/10/26.
//
#include "RequestHandlerFactory.hpp"
#include "WebSocketRequestHandler.hpp"

namespace {
class DefaultRequestHandler:public Poco::Net::HTTPRequestHandler{
  public:
    void handleRequest(Poco::Net::HTTPServerRequest& request,Poco::Net::HTTPServerResponse& response) override{

    }
};
}

VS_START
namespace remote
{
Poco::Net::HTTPRequestHandler *RequestHandlerFactory::createRequestHandler(const Poco::Net::HTTPServerRequest &request)
{
    if (request.getURI() != "/rpc")
    {
        return new DefaultRequestHandler();
    }
    return new WebSocketRequestHandler();
}
}
VS_END
