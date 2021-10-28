//
// Created by wyz on 2021/10/28.
//

#pragma once
#include <VolumeSlicer/export.hpp>
#include <functional>
VS_START

namespace remote{

/**
 * @brief using json serialize and deserialize to pass message in network
 */
class JsonRPCService{
  public:
    using Callback = std::function<void(uint8_t*,uint32_t)>;
    virtual void process_message(const uint8_t*,uint32_t,const Callback&) = 0;

};

}

VS_END