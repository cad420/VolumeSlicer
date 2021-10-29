//
// Created by wyz on 2021/10/29.
//
#pragma once
#include <thread>
#include <memory>
#include "RPC/MessageQueue.hpp"
#include <Poco/URI.h>
#include <Poco/Net/WebSocket.h>
VS_START
namespace remote{

class ManagerClient{
public:
    explicit ManagerClient(std::string address);

    ~ManagerClient();

    auto get_address() const;

    void register_worker();

    void shutdown();

private:
    std::string address;
    Poco::URI uri;
    std::thread work;
    std::unique_ptr<Poco::Net::WebSocket> ws;
    MessageQueue* message_queue;
};

}

VS_END
