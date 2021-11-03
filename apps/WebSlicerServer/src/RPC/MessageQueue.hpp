//
// Created by wyz on 2021/10/28.
//
#pragma once
#include "Service.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
VS_START
namespace remote{


class MessageQueue{
  public:
    using Callback = JsonRPCService::Callback ;
    struct Task{
        std::vector<uint8_t> message;
        Callback callback;
    };
    static void set_queue_type(const std::string& name);
    static auto get_queue_type()->std::string;
    static MessageQueue& get_instance();

    void add_message(const uint8_t* msg,uint32_t size,const Callback& callback);
  private:
    MessageQueue();

    ~MessageQueue();

    void process();

  private:
    static std::string name;
    bool running;
    std::mutex mtx;
    std::vector<std::thread> workers;
    std::condition_variable cv;
    std::queue<Task> tasks;
};

}
VS_END