//
// Created by wyz on 2021/10/28.
//
#pragma once
#include "DataModel/Slice.hpp"
#include "DataModel/Image.hpp"
#include "DataModel/Volume.hpp"
#include "Service/JsonRPCService.hpp"
#include "RPCMethod.hpp"
#include <functional>
#include <mutex>
#include <condition_variable>
#include <VolumeSlicer/volume.hpp>
VS_START
namespace remote{
class SliceRenderer{
  public:
    static const auto& GetSliceRenderer();
    static void Release();

  private:
    SliceRenderer();
    static bool occupied;
    static std::mutex mtx;
    static std::condition_variable cv;
};
class SliceService: public JsonRPCService{
  public:
    using Callback = JsonRPCService::Callback ;

    SliceService();

    void process_message(const uint8_t* message,uint32_t size,const Callback& callback) override;

  protected:
    //rpc method

    /**
     * @brief get volume information about
     */
    Volume get();

    /**
     * @brief
     * @param slice
     */
    Image render(Slice slice);

    /**
     * @brief get guide map for slice, using max lod down-sample volume to render
     * @param slice
     */
    Image map(Slice slice);

    /*
     * @brief get render and map results in frame
     */
    Frame render_frame(Slice slice);

  private:
    std::unique_ptr<RPCMethod> methods;

};

}
VS_END
