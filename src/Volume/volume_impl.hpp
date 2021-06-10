//
// Created by wyz on 2021/6/7.
//

#ifndef VOLUMESLICER_VOLUME_IMPL_HPP
#define VOLUMESLICER_VOLUME_IMPL_HPP

#include<vector>

#include<VolumeSlicer/volume.hpp>

VS_START

template<VolumeType type>
class VolumeImpl;

template<>
class VolumeImpl<VolumeType::Raw>: public Volume<VolumeType::Raw>{
public:
    VolumeImpl(std::vector<uint8_t>&& data):raw_volume_data(std::move(raw_volume_data)){};
    VolumeType GetVolumeType() const override{return VolumeType::Raw;}

    uint8_t* GetData() override{return nullptr;};
private:
    std::vector<uint8_t> raw_volume_data;
};

template<>
class VolumeImpl<VolumeType::Comp>: public Volume<VolumeType::Comp>{
public:
    VolumeImpl();
    VolumeType GetVolumeType() const override{return VolumeType::Comp;}
    VolumeBlock GetBlock(const std::array<uint32_t,4>&) noexcept override{return {};}

private:

};


class VolumeSamplerImpl: public VolumeSampler{
public:

    void Sample(const Slice& slice,uint8_t* data) override;

};

VS_END


#endif //VOLUMESLICER_VOLUME_IMPL_HPP
