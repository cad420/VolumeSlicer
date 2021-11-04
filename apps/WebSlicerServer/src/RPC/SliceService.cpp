//
// Created by wyz on 2021/10/28.
//
#include "SliceService.hpp"
#include "Dataset/CompVolume.hpp"
#include <Utils/logger.hpp>
#include <VolumeSlicer/volume_sampler.hpp>

VS_START
namespace remote
{
SliceService::SliceService() : methods(std::make_unique<RPCMethod>())
{
    methods->register_method("render", {"slice","d","depth","direction"}, RPCMethod::GetHandler(&SliceService::render, *this));
}
void SliceService::process_message(const uint8_t *message, uint32_t size, const SliceService::Callback &callback)
{


    uint8_t *response;
    size_t total;
    mpack_writer_t response_writer;
    try
    {
        mpack_tree_t tree;
        mpack_tree_init_data(&tree,reinterpret_cast<const char*>(message),size);
        mpack_tree_parse(&tree);
        if(mpack_tree_error(&tree)!=mpack_ok){
            throw std::runtime_error("UnPack error");
        }
        mpack_node_t root = mpack_tree_root(&tree);

        if(root.data->type != mpack_type_map ||
            !mpack_node_map_contains_cstr(root,"method") ||
            !mpack_node_map_contains_cstr(root,"params")){
            throw std::runtime_error("Invalid request error: no method or params");
        }

        auto method_node = mpack_node_map_cstr(root,"method");
        if(method_node.data->type != mpack_type_str){
            throw std::runtime_error("Invalid request error: method is not str");
        }
        auto method = std::string(mpack_node_str(method_node),mpack_node_strlen(method_node));

        auto param_node = mpack_node_map_cstr(root,"params");



        mpack_writer_init_growable(&response_writer, reinterpret_cast<char **>(&response),
                                   reinterpret_cast<size_t *>(&total));
        mpack_start_map(&response_writer, 1);

        methods->invoke(method,param_node,&response_writer);
    }
    catch (const std::exception& err)
    {
        LOG_ERROR(err.what());
        mpack_writer_init_growable(&response_writer, reinterpret_cast<char **>(&response),
                                   reinterpret_cast<size_t *>(&total));
        mpack_start_map(&response_writer, 1);
        mpack_write_cstr(&response_writer,"error");
        mpack_write_cstr(&response_writer,err.what());
    }


    mpack_finish_map(&response_writer);
    if(mpack_writer_destroy(&response_writer)==mpack_ok){
        callback(reinterpret_cast<const uint8_t*>(response),total);
    }
    free(response);
    LOG_INFO("finish");
}
//-----------------------------------------------------

bool SliceRenderer::occupied = false;
std::mutex SliceRenderer::mtx;
std::condition_variable SliceRenderer::cv;

const auto& SliceRenderer::GetSliceRenderer()
{
    static SliceRenderer sliceRenderer;

    static auto slice_sampler = VolumeSampler::CreateVolumeSampler(
        VolumeDataSet::GetVolume()
    );
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk,[](){
        return !occupied;
    });
    occupied = true;
    return slice_sampler;
}

void SliceRenderer::Release()
{
    occupied = false;
    cv.notify_one();
}
SliceRenderer::SliceRenderer()
{
    SetCUDACtx(0);
}

static void MaxMix(std::vector<uint8_t>& res,std::vector<uint8_t>& v){
    assert(res.size()==v.size());
    for(size_t i = 0;i<res.size();i++){
        res[i] = (std::max)(res[i],v[i]);
    }
}

//rpc method
Image SliceService::render(Slice slice,int d,float depth,int direction)
{
    cuCtxSetCurrent(GetCUDACtx());
    auto slicer = Slicer::CreateSlicer(slice);
    slicer->SetSliceSpaceRatio({1.f,1.f,1.f/0.32f});
    const auto& slice_renderer = SliceRenderer::GetSliceRenderer();
    std::vector<uint8_t> res(slice.n_pixels_width*slice.n_pixels_height);
    slice_renderer->Sample(slice,res.data(),false);
    std::vector<uint8_t> tmp;
    if(d>0){
        tmp.resize(slice.n_pixels_width*slice.n_pixels_height);
    }
    if((direction & 0b1)){
        for(int i=1;i<=d;i++){
            slicer->MoveByNormal(depth);
            slice_renderer->Sample(slicer->GetSlice(),tmp.data(),false);
            MaxMix(res,tmp);
        }
    }
    slicer->SetSlice(slice);
    if((direction & 0b10)){

        for(int i=1;i<=d;i++){
            slicer->MoveByNormal(-depth);
            slice_renderer->Sample(slicer->GetSlice(),tmp.data(),false);
            MaxMix(res,tmp);
        }
    }
    auto img = Image::encode(res.data(),slice.n_pixels_width,slice.n_pixels_height,1,
                             Image::Format::JPEG,Image::Quality::MEDIUM);
    SliceRenderer::Release();
    return img;
}


}
VS_END
