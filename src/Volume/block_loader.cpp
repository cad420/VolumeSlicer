//
// Created by wyz on 2021/6/9.
//
#include"Volume/block_loader.hpp"
#include<VoxelCompression/voxel_uncompress/VoxelUncompress.h>
#include<spdlog/spdlog.h>
VS_START
class Worker{
public:
    Worker(const VoxelUncompressOptions& opt){
        uncmp=std::make_unique<VoxelUncompress>(opt);
        status._a=false;
    }
    bool isBusy() const{
        return status._a;
    }
    void setStatus(bool _status){
        status._a = _status;
    }
    void uncompress(uint8_t* dest_ptr,int64_t len,std::vector<std::vector<uint8_t>>& packets){
        uncmp->uncompress(dest_ptr,len,packets);
    }
private:
    std::unique_ptr<VoxelUncompress> uncmp;
    atomic_wrapper<bool> status;
};

BlockLoader::BlockLoader()
:block_size_bytes(0),cu_mem_num(16),worker_num(2)
{

}
void BlockLoader::Open(const std::string& filename)
{
    this->packet_reader=Reader::CreateReader(filename.c_str());
    //!only after create reader then can know block's information
    this->block_size_bytes=packet_reader->GetBlockSizeByte();
    spdlog::info("block_size_bytes is: {0}.",block_size_bytes);
    this->cu_mem_pool=std::make_unique<CUDAMemoryPool<uint8_t>>(cu_mem_num,block_size_bytes);

    VoxelUncompressOptions uncmp_opts;
    auto frame_shape=packet_reader->GetFrameShape();
    uncmp_opts.width=frame_shape[0];
    uncmp_opts.height=frame_shape[1];
    uncmp_opts.use_device_frame_buffer=true;
    uncmp_opts.cu_ctx=GetCUDACtx();
    for(int i=0;i<worker_num;i++)
        workers.emplace_back(uncmp_opts);

    jobs=std::make_unique<ThreadPool>(worker_num);

    products.setSize(cu_mem_num*2);//max is cu_mem_num
}

size_t BlockLoader::GetAvailableNum() {
    size_t num=0;
    for(auto& worker:workers){
        if(!worker.isBusy()){
            num++;
        }
    }
//    spdlog::info("valid cu_mem num: {0}.",cu_mem_pool->GetValidCUDAMemNum());
    return num;
}

bool BlockLoader::AddTask(const std::array<uint32_t, 4> &idx) {
    //check if idx is valid
    if(idx[0]==INVALID || idx[1]==INVALID || idx[2]==INVALID || idx[3]==INVALID){
        return false;
    }

    if(GetAvailableNum()==0){
        return false;
    }
    else{
        for(size_t i=0;i<workers.size();i++){
            if(!workers[i].isBusy()){
                workers[i].setStatus(true);
                spdlog::info("worker {0} append task.",i);
                jobs->AppendTask([&](int worker_id,const std::array<uint32_t,4>& idx){
                    std::vector<std::vector<uint8_t>> packet;
                    packet_reader->GetPacket(idx,packet);
                    VolumeBlock block;
                    block.index=idx;
                    spdlog::info("in AppendTask {0} {1} {2} {3}.",block.index[0],block.index[1],block.index[2],block.index[3]);

//                    spdlog::info("before cu_mem_pool valid cu_mem num: {0}.",cu_mem_pool->GetValidCUDAMemNum());
                    block.block_data=cu_mem_pool->GetCUDAMem();
//                    spdlog::info("after cu_mem_pool valid cu_mem num: {0}.",cu_mem_pool->GetValidCUDAMemNum());
//                    spdlog::info("start uncompress");
//                    START_CPU_TIMER
                    assert(block.block_data->GetDataPtr());
                    workers[worker_id].uncompress(block.block_data->GetDataPtr(),block_size_bytes,packet);
//                    END_CPU_TIMER
//                    spdlog::info("finish uncompress");
                    block.valid=true;
                    products.push_back(block);
//                    spdlog::info("products size: {0}.",products.size());
                    workers[worker_id].setStatus(false);
//                    spdlog::info("finish one job");
                },i,idx);
                break;
            }
        }
        return true;
    }
}

bool BlockLoader::IsEmpty() {
    return products.empty();
}

auto BlockLoader::GetBlock() -> Volume<VolumeType::Comp>::VolumeBlock {
    if(IsEmpty()){
        VolumeBlock block;
        block.block_data=nullptr;
        block.valid=false;
        block.index={INVALID,INVALID,INVALID,INVALID};
        return block;
    }
    else{
        spdlog::info("before GetBlock, products size: {0}.",products.size());
        return products.pop_front();
    }
}
//!!!!!!
BlockLoader::~BlockLoader() {
    //! must destruct jobs first
    jobs.reset();
    spdlog::info("Delete block_loader...Remain product num: {0}.",products.size());
    workers.clear();
    products.clear();
    packet_reader.reset();
    cu_mem_pool.reset();
}

auto BlockLoader::GetBlockDim(int lod) const -> std::array<uint32_t, 3> {
    return packet_reader->GetBlockDim(lod);
}

auto BlockLoader::GetBlockLength() const -> std::array<uint32_t, 4> {
    return packet_reader->GetBlockLength();
}

bool BlockLoader::IsAllAvailable() {
    return GetAvailableNum()==worker_num;
}

VS_END

