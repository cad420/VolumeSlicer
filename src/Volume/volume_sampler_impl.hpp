//
// Created by wyz on 2021/6/11.
//

#ifndef VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP
#define VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP

#include<VolumeSlicer/volume.hpp>

VS_START

class VolumeSamplerImpl: public VolumeSampler{
public:


    void Sample(const Slice& slice,uint8_t* data) override;

private:
    void SampleRaw(const Slice& slice,uint8_t* data);

    void SampleComp(const Slice& slice,uint8_t* data);

private:
    std::shared_ptr<VolumeBase> volume;

};




VS_END


#endif //VOLUMESLICER_VOLUME_SAMPLER_IMPL_HPP
