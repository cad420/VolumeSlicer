//
// Created by wyz on 2021/10/28.
//
#pragma once
#include "DataModel/Slice.hpp"
#include "DataModel/Image.hpp"
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
     * @brief
     * @param slice
     * @param d number of slice should be sampled except the central slice
     * @param depth voxels between two slices
     * @param direction 1 represent forward slice's normal, 2 represent backward slice's normal, 3 represent double direction
     */
    Image render(Slice slice,float depth,int direction);

    /**
     * @brief get guide map for slice, using max lod down-sample volume to render
     * @param slice
     */
    Image map(Slice slice,int window_w,int window_h);

  private:
    std::unique_ptr<RPCMethod> methods;

};

}
VS_END
