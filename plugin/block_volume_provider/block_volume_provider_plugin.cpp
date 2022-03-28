//
// Created by wyz on 2021/10/8.
//
#include "block_volume_provider_plugin.hpp"

#include <VolumeSlicer/Utils/logger.hpp>
#include <VolumeSlicer/Utils/plugin_loader.hpp>

//#include <VoxelCompression/voxel_uncompress/VoxelUncompress.h>
#include "volume_transform.cuh"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

VS_START

struct Decoder{
    size_t decode(AVCodecContext* c,AVFrame* frame,AVPacket* pkt,uint8_t* buf){
        int ret = avcodec_send_packet(c,pkt);
        if(ret < 0){
            throw std::runtime_error("error sending a packet for decoding");
        }
        size_t frame_pos = 0;
        while(ret >= 0){
            ret = avcodec_receive_frame(c,frame);
            if(ret == AVERROR(EAGAIN) || ret ==AVERROR_EOF)
                break;
            else if(ret < 0){
                throw std::runtime_error("error during decoding");
            }
            memcpy(buf + frame_pos,frame->data[0],frame->linesize[0]*frame->height);
            frame_pos += frame->linesize[0] * frame->height;
        }
        return frame_pos;
    }

    void uncompress(uint8_t* data,size_t len,std::vector<std::vector<uint8_t>>& packets){
        auto codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        assert(codec);
//        auto parser = av_parser_init(codec->id);
//        assert(parser);

        auto c = avcodec_alloc_context3(codec);
        c->thread_count = 16;
        c->delay = 0;
        assert(c);
        int ret = avcodec_open2(c,codec,nullptr);
        assert(ret >= 0);
        auto frame = av_frame_alloc();
        assert(frame);
        auto pkt = av_packet_alloc();
        assert(pkt);
        uint8_t* p = data;
        size_t offset = 0;

        for(auto& packet:packets){
            pkt->data = packet.data();
            pkt->size = packet.size();
            offset += decode(c,frame,pkt,p+offset);
            if(offset > len){
                throw std::runtime_error("decode result out of buffer range");
            }
        }
        decode(c,frame,nullptr,p+offset);

        avcodec_free_context(&c);
        av_frame_free(&frame);
        av_packet_free(&pkt);
    }
};

class Worker
{
  public:
    Worker()
    {
        uncmp = std::make_unique<Decoder>();
        status._a = false;
    }
    bool isBusy() const
    {
        return status._a;
    }
    void setStatus(bool _status)
    {
        status._a = _status;
    }
    void uncompress(uint8_t *dest_ptr, int64_t len, std::vector<std::vector<uint8_t>> &packets)
    {
        START_CPU_TIMER
        uncmp->uncompress(dest_ptr, len, packets);
        END_CPU_TIMER
    }

  private:
    std::unique_ptr<Decoder> uncmp;
    atomic_wrapper<bool> status;
};

BlockVolumeProviderPlugin::BlockVolumeProviderPlugin() : block_size_bytes(0), cu_mem_num(16), worker_num(4)
{
    SetCUDACtx(0);
    PluginLoader::LoadPlugins("./plugins");
    this->packet_reader = std::unique_ptr<IH264VolumeReaderPluginInterface>(
        PluginLoader::CreatePlugin<IH264VolumeReaderPluginInterface>(".h264"));
    if (!packet_reader)
    {
        throw std::runtime_error("IH264VolumeReaderPlugin load failed.");
    }
    LOG_INFO("Create plugin for h264 read.");
}
BlockVolumeProviderPlugin::~BlockVolumeProviderPlugin()
{
    //! must destruct jobs first
    jobs.reset();
    LOG_INFO("Delete block_loader...Remain product num: {0}.", products.size());
    workers.clear();
    products.clear();
    packet_reader.reset();
    cu_mem_pool.reset();
}
void BlockVolumeProviderPlugin::Open(const std::string &filename)
{
    this->packet_reader->Open(filename);

    //! only after create reader then can know block's information
    this->block_size_bytes = packet_reader->GetBlockSizeByte();
    auto block_length = packet_reader->GetBlockLength()[0];
    this->block_voxel_count = block_length * block_length * block_length;
    this->voxel_size = this->block_size_bytes / this->block_voxel_count;
    LOG_INFO("block_size_bytes is: {0}.", block_size_bytes);
    LOG_INFO("voxel size: {0}",voxel_size);
    this->cu_mem_pool = std::make_unique<CUDAMemoryPool<uint8_t>>(cu_mem_num, block_voxel_count);
//    this->decode_mem_pool = std::make_unique<CUDAMemoryPool<uint8_t>>(worker_num,block_size_bytes);


    for (int i = 0; i < worker_num; i++)
        workers.emplace_back();

    jobs = std::make_unique<ThreadPool>(worker_num);

    products.setSize(cu_mem_num * 2); // max is cu_mem_num
}

auto BlockVolumeProviderPlugin::GetBlockDim(int lod) const -> std::array<uint32_t, 3>
{
    return packet_reader->GetBlockDim(lod);
}

auto BlockVolumeProviderPlugin::GetBlockLength() const -> std::array<uint32_t, 4>
{
    return packet_reader->GetBlockLength();
}

bool BlockVolumeProviderPlugin::AddTask(const std::array<uint32_t, 4> &idx)
{
    // check if idx is valid
    if (idx[0] == INVALID || idx[1] == INVALID || idx[2] == INVALID || idx[3] == INVALID)
    {
        return false;
    }

    if (GetAvailableNum() == 0)
    {
        return false;
    }
    else
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (size_t i = 0; i < workers.size(); i++)
        {

            if (!workers[i].isBusy())
            {
                workers[i].setStatus(true);
                LOG_INFO("worker {0} append task.", i);
                jobs->AppendTask(
                    [&](int worker_id, const std::array<uint32_t, 4> &idx) {
                        std::vector<std::vector<uint8_t>> packet;
                        packet_reader->GetPacket(idx, packet);
                        VolumeBlock block;
                        block.index = idx;
                        LOG_INFO("in AppendTask {0} {1} {2} {3}.", block.index[0], block.index[1], block.index[2],
                                 block.index[3]);

                        std::vector<uint8_t> decode_buffer(block_size_bytes,0);

                        std::vector<uint8_t> decode_uint8_buffer(block_voxel_count,0);

                        workers[worker_id].uncompress(decode_buffer.data(), block_size_bytes, packet);

                        block.block_data = cu_mem_pool->GetCUDAMem();
                        assert(block.block_data->GetDataPtr());

                        if(voxel_size == 1){
                          decode_uint8_buffer = std::move(decode_buffer);
                        }
                        else if(voxel_size == 2){
                            using VoxelT = uint16_t;
                            auto p = (uint16_t*)decode_buffer.data();
                            uint16_t max_v = 0,min_v = std::numeric_limits<uint16_t>::max();
                            for(size_t i = 0;i<block_voxel_count;i++){
                                if(p[i] < min_v) min_v = p[i];
                                if(p[i] > max_v) max_v = p[i];
                            }
                            float inv;
                            if(max_v!=min_v)
                                inv = 255.f/(max_v - min_v);
                            else inv = 0.f;
                            for(size_t i = 0;i < block_voxel_count;i++){
                                decode_uint8_buffer[i] = (p[i] - min_v) * inv;
                            }
                        }
                        else{
                            throw std::runtime_error("not support voxel size now");
                        }
                        cuCtxSetCurrent(GetCUDACtx());
                        CUDA_DRIVER_API_CALL(cuMemcpy((CUdeviceptr)block.block_data->GetDataPtr(),(CUdeviceptr)decode_uint8_buffer.data(),decode_uint8_buffer.size()));
//                        if(voxel_size == 2){
//                            BitTransformToUInt8(decode_buffer->GetDataPtr(),block.block_data->GetDataPtr(),block_voxel_count,BitTransformDataType::uint16);
//                            LOG_INFO("decoded buffer from uint16 ==> uint8");
//                        }
//                        else if(voxel_size == 1){
//                            BitTransformToUInt8(decode_buffer->GetDataPtr(),block.block_data->GetDataPtr(),block_voxel_count,BitTransformDataType::uint8);
//                            LOG_INFO("decoded buffer from uint8 ==> uint8");
//                        }
//                        else{
//                            throw std::runtime_error("not support voxel size now");
//                        }

                        block.valid = true;
                        products.push_back(block);

                        workers[worker_id].setStatus(false);
                        LOG_INFO("worker {} finish job",worker_id);
                    },
                    i, idx);
                break;
            }
        }
        return true;
    }
}

auto BlockVolumeProviderPlugin::GetBlock() -> CompVolume::VolumeBlock
{
    if (IsEmpty())
    {
        VolumeBlock block;
        block.block_data = nullptr;
        block.valid = false;
        block.index = {INVALID, INVALID, INVALID, INVALID};
        return block;
    }
    else
    {
        LOG_INFO("before GetBlock, products size: {0}.", products.size());
        return products.pop_front();
    }
}

bool BlockVolumeProviderPlugin::IsEmpty()
{
    return products.empty();
}

size_t BlockVolumeProviderPlugin::GetAvailableNum()
{
    size_t num = 0;
    for (auto &worker : workers)
    {
        if (!worker.isBusy())
        {
            num++;
        }
    }
    return num;
}

bool BlockVolumeProviderPlugin::IsAllAvailable()
{
    return GetAvailableNum() == worker_num;
}
auto BlockVolumeProviderPlugin::GetVolumeSpace() const -> std::array<float, 3>
{
    return packet_reader->GetVolumeSpace();
}

VS_END

VS_REGISTER_PLUGIN_FACTORY_IMPL(BlockVolumeProviderPluginFactory)
EXPORT_PLUGIN_FACTORY_IMPL(BlockVolumeProviderPluginFactory)