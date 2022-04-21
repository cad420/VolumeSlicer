//
// Created by wyz on 2022/4/21.
//
#pragma once
#include <VolumeSlicer/Data/volume.hpp>
VS_START

class CompVolumeAdapter{
  public:
    void ClearRequestBlock(CompVolume* comp_volume) noexcept {
        comp_volume->ClearRequestBlock();
    }

    static void SetRequestBlock(CompVolume* comp_volume,const std::array<uint32_t,4>& requests) noexcept {
        comp_volume->SetRequestBlock(requests);
    }

    static void EraseBlockInRequest(CompVolume* comp_volume,const std::array<uint32_t,4>& requests) noexcept {
        comp_volume->EraseBlockInRequest(requests);
    }

    static void ClearBlockQueue(CompVolume* comp_volume) noexcept {
        comp_volume->ClearBlockQueue();
    }

    static void ClearBlockInQueue(CompVolume* comp_volume,const std::vector<std::array<uint32_t,4>>& targets) noexcept {
        comp_volume->ClearBlockInQueue(targets);
    }

    static void ClearAllBlockInQueue(CompVolume* comp_volume) noexcept {
        comp_volume->ClearAllBlockInQueue();
    }

    static int GetBlockQueueSize(CompVolume* comp_volume) {
        return comp_volume->GetBlockQueueSize();
    }

    static int GetBlockQueueMaxSize(CompVolume* comp_volume) {
        return comp_volume->GetBlockQueueMaxSize();
    }

    static void SetBlockQueueSize(CompVolume* comp_volume,size_t count) {
        comp_volume->SetBlockQueueSize(count);
    }

    static void PauseLoadBlock(CompVolume* comp_volume) noexcept {
        comp_volume->PauseLoadBlock();
    }

    static void StartLoadBlock(CompVolume* comp_volume) noexcept {
        comp_volume->StartLoadBlock();
    }

    static bool GetStatus(CompVolume* comp_volume) {
        return comp_volume->GetStatus();
    }
};

VS_END