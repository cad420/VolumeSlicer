//
// Created by wyz on 2021/6/11.
//

#ifndef VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP
#define VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP

#include<VolumeSlicer/volume_sampler.hpp>

VS_START

template<class T>
class VolumeSamplerImpl;

class CUDARawVolumeSampler;

template<>
class VolumeSamplerImpl<RawVolume>: public VolumeSampler{
public:
    explicit VolumeSamplerImpl(const std::shared_ptr<RawVolume>& volume);

    void Sample(const Slice& slice,uint8_t* data) override;

private:

private:
    std::shared_ptr<RawVolume> raw_volume;
    std::unique_ptr<CUDARawVolumeSampler> cuda_raw_volume_sampler;
};



template<>
class VolumeSamplerImpl<CompVolume>: public VolumeSampler{
public:
    explicit VolumeSamplerImpl(const std::shared_ptr<CompVolume>& volume):comp_volume(volume){};

    void Sample(const Slice& slice,uint8_t* data) override;


private:
    std::shared_ptr<CompVolume> comp_volume;
};





VS_END


#endif //VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP
