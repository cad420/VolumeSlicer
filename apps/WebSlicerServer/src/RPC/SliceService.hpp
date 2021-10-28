//
// Created by wyz on 2021/10/28.
//
#pragma once
#include "RPCMethod.hpp"
#include <functional>
#include "Service/JsonRPCService.hpp"
VS_START
namespace remote{

class SliceService: public JsonRPCService{
  public:
    using Callback = JsonRPCService::Callback ;

    SliceService();

    void process_message(const uint8_t* message,uint32_t size,const Callback& callback) override;


};

}
VS_END
