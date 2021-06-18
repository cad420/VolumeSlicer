//
// Created by wyz on 2021/6/11.
//
#include"Algorithm/volume_sampler_impl.hpp"
#include"Algorithm/raw_volume_sample.cuh"
VS_START

/**************************************************************************************************
 * API for RawVolumeSamplerImpl
 */

std::unique_ptr<VolumeSampler> vs::VolumeSampler::CreateVolumeSampler(const std::shared_ptr<RawVolume> & volume) {
    return std::make_unique<VolumeSamplerImpl<RawVolume>>(volume);
}
VolumeSamplerImpl<RawVolume>::VolumeSamplerImpl(const std::shared_ptr<RawVolume> &volume):raw_volume(volume){
    this->cuda_raw_volume_sampler=std::make_unique<CUDARawVolumeSampler>();
    this->cuda_raw_volume_sampler->SetVolumeData(raw_volume->GetData(),
                                                 raw_volume->GetVolumeDimX(),
                                                 raw_volume->GetVolumeDimY(),
                                                 raw_volume->GetVolumeDimZ());
    spdlog::info("Successfully create raw volume sampler.");
}

void VolumeSamplerImpl<RawVolume>::Sample(const Slice &slice, uint8_t *data) {
    cuda_raw_volume_sampler->sample(data,slice,
                                    raw_volume->GetVolumeSpaceX(),
                                    raw_volume->GetVolumeSpaceY(),
                                    raw_volume->GetVolumeSpaceZ());
}





/**************************************************************************************************
 * API for CompVolumeSamplerImpl
 */

std::unique_ptr<VolumeSampler> VolumeSampler::CreateVolumeSampler(const std::shared_ptr<CompVolume> & volume) {
    return std::make_unique<VolumeSamplerImpl<CompVolume>>(volume);
}

void VolumeSamplerImpl<CompVolume>::Sample(const Slice &slice, uint8_t *data) {

}



VS_END
