//
// Created by wyz on 2021/6/18.
//

#ifndef VOLUMESLICER_VOLUME_SAMPLER_HPP
#define VOLUMESLICER_VOLUME_SAMPLER_HPP

#include<VolumeSlicer/volume.hpp>

VS_START


class VS_EXPORT VolumeSampler{
public:
    VolumeSampler()=default;

    virtual ~VolumeSampler(){}

    static std::unique_ptr<VolumeSampler> CreateVolumeSampler(const std::shared_ptr<RawVolume>&);

    static std::unique_ptr<VolumeSampler> CreateVolumeSampler(const std::shared_ptr<CompVolume>&);

    //data should has be alloc and its size equal to slice.n_pixels_width * slice.n_pixels_height
    //data could be cuda device ptr or cpu host ptr
    //volume's properties are according to CreateVolumeSampler's shared_ptr volume
    virtual bool Sample(const Slice& slice,uint8_t* data)=0;

};

VS_END

#endif //VOLUMESLICER_VOLUME_SAMPLER_HPP
