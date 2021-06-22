//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_VOLUME_IMPL_HPP
#define VOLUMESLICER_VOLUME_IMPL_HPP

#include<vector>

#include<VolumeSlicer/volume_sampler.hpp>
#include<VolumeSlicer/utils.hpp>


VS_START

class BlockLoader;

template<VolumeType type>
class VolumeImpl;

template<>
class VolumeImpl<VolumeType::Raw>: public Volume<VolumeType::Raw>{
public:
    VolumeImpl(std::vector<uint8_t>&& data):raw_volume_data(std::move(data)){};
    VolumeType GetVolumeType() const override{return VolumeType::Raw;}

    uint8_t* GetData() override{return raw_volume_data.data();};
private:
    std::vector<uint8_t> raw_volume_data;
};
using RawVolumeImpl=VolumeImpl<VolumeType::Raw>;

template<>
class VolumeImpl<VolumeType::Comp>: public Volume<VolumeType::Comp>{
public:
    explicit VolumeImpl(const char* file_name);

    //base class must define ~ function
    ~VolumeImpl();

    VolumeType GetVolumeType() const override{return VolumeType::Comp;}

    void ClearRequestBlock() noexcept override;

    void SetRequestBlock(const std::array<uint32_t,4>&) noexcept override;

    void EraseBlockInRequest(const std::array<uint32_t,4>&) noexcept override;

    void ClearBlockQueue() noexcept override;

    void ClearAllBlockInQueue() noexcept override;

    int GetBlockQueueSize() override;

    void PauseLoadBlock() noexcept override;

    void StartLoadBlock() noexcept override;

    VolumeBlock GetBlock(const std::array<uint32_t,4>&) noexcept override;

    auto GetBlockDim(int lod) const ->std::array<uint32_t ,3> override;

    auto GetBlockLength() const ->std::array<uint32_t,2> override;

private:
    bool FindInRequestBlock(const std::array<uint32_t,4>& idx);

    bool FindInBlockQueue(const std::array<uint32_t,4>& idx) const=delete;

    //no blocking
    auto FetchRequest()->std::array<uint32_t,4>;

    void AddBlocks();

    void Loading();
private:
    //while clear block_queue, can't add block to the queue.
    //while operate on request_queue, loader can't operate
    std::mutex mtx;

    std::condition_variable cv;
    bool pause;
    bool stop;

    std::unique_ptr<BlockLoader> block_loader;

    std::thread task;

    std::list<std::array<uint32_t,4>> request_queue;

    ConcurrentQueue<VolumeBlock> block_queue;

};
using CompVolumeImpl=VolumeImpl<VolumeType::Comp>;



VS_END


#endif //VOLUMESLICER_VOLUME_IMPL_HPP
