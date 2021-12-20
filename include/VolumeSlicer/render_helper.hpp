//
// Created by wyz on 2021/12/20.
//

#pragma once
#include <VolumeSlicer/render.hpp>
#include <VolumeSlicer/Utils/logger.hpp>
#include <cmath>
VS_START

struct CompRenderHelper{
    /**
     * @brief Base lod pocily means each lod distance is minimum for keep real-time render performance and quality.
     * This is calculated by block length and volume space.
     * @param lod_policy will resize to 10 and fill with result.
     */
    static void GetBaseLodPolicy(const std::shared_ptr<CompVolume>& comp_volume,std::vector<double>& lod_policy){
        auto volume_space_x = comp_volume->GetVolumeSpaceX();
        auto volume_space_y = comp_volume->GetVolumeSpaceY();
        auto volume_space_z = comp_volume->GetVolumeSpaceZ();
        auto base_space = std::sqrt(volume_space_x*volume_space_x+volume_space_y*volume_space_y+volume_space_z*volume_space_z);
        auto block_length = comp_volume->GetBlockLength()[0];
        auto base_lod = base_space * (block_length + 2) / 2 ;
        lod_policy.resize(10,0.f);
        for(int i=0;i<9;i++){
            lod_policy[i] = base_lod * std::pow(2,i);
        }
        lod_policy[9] = std::numeric_limits<double>::max();
    }

    /**
     * @brief Get default view pos for camera to see the comp-volume. Default pos is measured in volume space.
     */
    static void GetDefaultViewPos(const std::shared_ptr<CompVolume>& comp_volume,std::array<float,3>& view_pos){
        auto volume_space_x = comp_volume->GetVolumeSpaceX();
        auto volume_space_y = comp_volume->GetVolumeSpaceY();
        auto volume_space_z = comp_volume->GetVolumeSpaceZ();
        auto volume_dim_x = comp_volume->GetVolumeDimX();
        auto volume_dim_y = comp_volume->GetVolumeDimY();
        auto volume_dim_z = comp_volume->GetVolumeDimZ();
        auto volume_board_x = volume_space_x * volume_dim_x;
        auto volume_board_y = volume_space_y * volume_dim_y;
        auto volume_board_z = volume_space_z * volume_dim_z;
        view_pos = {volume_board_x*0.5f,volume_board_y*0.5f,volume_board_z * 0.95f};
    }

    /**
     * @brief Get default suitable step and steps for volume ray-cast render.
     */
    static void GetDefaultRayCastStep(const std::shared_ptr<CompVolume>& comp_volume,float& step,int& steps){
        auto volume_space_x = comp_volume->GetVolumeSpaceX();
        auto volume_space_y = comp_volume->GetVolumeSpaceY();
        auto volume_space_z = comp_volume->GetVolumeSpaceZ();
        auto volume_dim_x = comp_volume->GetVolumeDimX();
        auto volume_dim_y = comp_volume->GetVolumeDimY();
        auto volume_dim_z = comp_volume->GetVolumeDimZ();
        auto volume_board_x = volume_space_x * volume_dim_x;
        auto volume_board_y = volume_space_y * volume_dim_y;
        auto volume_board_z = volume_space_z * volume_dim_z;
        auto base_space = std::min({volume_space_x,volume_space_y,volume_space_z});
        step = 0.5f * base_space;
        auto base_board = std::max({volume_board_x,volume_board_y,volume_board_z});
        steps = 0.2f * base_board / base_space;
        LOG_INFO("GetDefaultRayCastStep: step({}), steps({}).",step,steps);
    }
};

VS_END