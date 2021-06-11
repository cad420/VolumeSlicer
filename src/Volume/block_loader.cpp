//
// Created by wyz on 2021/6/9.
//
#include"Volume/block_loader.hpp"
#include<VoxelCompression/voxel_uncompress/VoxelUncompress.h>

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
        status._a=_status;
    }
    void uncompress(uint8_t* dest_ptr,int64_t len,std::vector<std::vector<uint8_t>>& packets){
        uncmp->uncompress(dest_ptr,len,packets);
    }
private:
    std::unique_ptr<VoxelUncompress> uncmp;
    atomic_wrapper<bool> status;
};

size_t BlockLoader::GetAvailableNum() {
    size_t num=0;
    for(auto& worker:workers){
        if(!worker.isBusy()){
            num++;
        }
    }
    return num;
}

void BlockLoader::AddTask(const std::array<uint32_t, 4> &idx) {
    if(GetAvailableNum()==0){
        return ;
    }
    else{
        for(size_t i=0;i<workers.size();i++){
            if(!workers[i].isBusy()){
                workers[i].setStatus(true);

                jobs->AppendTask([&](int worker_id,const std::array<uint32_t,4>& idx){
                    std::vector<std::vector<uint8_t>> packet;
                    packet_reader->GetPacket(idx,packet);
                    VolumeBlock block;
                    block.index=idx;
                    block.block_data=cu_mem_pool->GetCUDAMem();
                    workers[worker_id].uncompress(block.block_data->GetDataPtr(),block_size_bytes,packet);
                    block.valid=true;
                    products.push_back(block);
                    workers[worker_id].setStatus(false);

                },i,idx);
            }
        }
    }
}

bool BlockLoader::IsEmpty() {
    return products.empty();
}

auto BlockLoader::GetBlock() -> Volume<VolumeType::Comp>::VolumeBlock {
    return Volume<VolumeType::Comp>::VolumeBlock();
}


VS_END

