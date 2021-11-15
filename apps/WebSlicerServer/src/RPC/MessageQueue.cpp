//
// Created by wyz on 2021/10/28.
//
#include "MessageQueue.hpp"
#include <VolumeSlicer/Utils/logger.hpp>
#include <iostream>
VS_START
namespace remote
{
std::string MessageQueue::name;
MessageQueue::MessageQueue()
{
    running = true;
    for(int i=0;i<std::thread::hardware_concurrency();i++){
        workers.emplace_back(&MessageQueue::process,this);
    }
}

MessageQueue::~MessageQueue()
{
    if(!running){
        return;
    }
    running = false;
    cv.notify_all();
    for(auto& t:workers){
        if(t.joinable()){
            t.join();
        }
    }
}

MessageQueue &MessageQueue::get_instance()
{
    if(name.empty()){
        throw std::runtime_error("MessageQueue type name not initialize before get instance");
    }
    static MessageQueue message_queue{};
    return message_queue;
}

void MessageQueue::add_message(const uint8_t *msg, uint32_t size,const Callback& callback)
{
    std::unique_lock<std::mutex> lk(mtx);
    MessageQueue::Task task{std::vector<uint8_t>(msg,msg+size),callback};
    tasks.emplace(std::move(task));
    cv.notify_one();
}

void MessageQueue::process()
{
    auto service = CreateServiceByName(name);

    while(running){
        std::unique_lock<std::mutex> lk(mtx);
        if(!tasks.empty()){
            std::cout<<"Message queue worker thread id: "<<std::this_thread::get_id()<<std::endl;
            auto task = std::move(tasks.front());
            tasks.pop();
            lk.unlock();
            //todo lock for callback?
            service->process_message(task.message.data(),task.message.size(),task.callback);
        }
        else if(running){
            cv.wait(lk);
            lk.unlock();
        }

    }
}
void MessageQueue::set_queue_type(const std::string &type_name)
{
    name = type_name;
}
auto MessageQueue::get_queue_type()->std::string
{
    return name;
}

}
VS_END
