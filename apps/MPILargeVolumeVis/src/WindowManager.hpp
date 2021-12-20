//
// Created by wyz on 2021/7/12.
//

#pragma once
#include <json.hpp>
#include <fstream>
#include <mpi.h>
#include <iostream>
#include <optional>
using nlohmann::json;
class WindowManager{
public:
    struct AdvanceOptions{
        struct LodPolicy{
            std::optional<std::vector<double>> lod_dist;
            std::optional<std::string> cdf_value_file;
        }lod_policy;
        struct RayCast{
            std::optional<int> steps;
            std::optional<float> step;
        }ray_cast;
    };
    WindowManager(const char* config_file){
        std::ifstream in(config_file);
        if(!in.is_open()){
            throw std::runtime_error("Open config file failed.");
        }
        json j;
        in>>j;
        in.close();

        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Get_processor_name(processName,&processNameLength);
        std::cout<<"world size is: "<<world_size<<", node "<<world_rank<<" is on host "<<processName<<std::endl;

        world_window_width=j.at("width");
        world_window_height=j.at("height");
        row_num=j.at("row");
        col_num=j.at("col");
        frame_time_lock = j.at("frame_time_lock");
        renderer_backend = j.at("renderer_backend");
        iGPU = j.at("iGPU");
        {
            auto space = j.at("space");
            space_x = space.at(0);
            space_y = space.at(1);
            space_z = space.at(2);
        }

        //read tf
        {
            auto tf = j.at("tf");
            for(auto it=tf.begin();it!=tf.end();it++){
                int key = std::stoi(it.key());
                std::array<double,4> value = it.value();
                tf_map[key] = value;
            }
        }

        //read advance options
        {
            try{
                auto opts = j.at("advance_options");
                std::vector<double> lod_dist = opts.at("lod_policy").at("lod_dist");
                std::string cdf_value_file = opts.at("lod_policy").at("cdf_value_file");
                if(!lod_dist.empty()){
                    for(auto& d:lod_dist){
                        if(d==-1){
                            d = std::numeric_limits<double>::max();
                        }
                    }
                    advance_options.lod_policy.lod_dist = lod_dist;
                }
                if(!cdf_value_file.empty()){
                    advance_options.lod_policy.cdf_value_file = cdf_value_file;
                }
                int steps = opts.at("ray_cast").at("steps");
                float step = opts.at("ray_cast").at("step");
                if(steps>0){
                    advance_options.ray_cast.steps = steps;
                    LOG_INFO("Config advance options: steps({}).",steps);
                }
                if(step>0.f){
                    advance_options.ray_cast.step = step;
                    LOG_INFO("Config advance options: step({}).",step);
                }
            }
            catch (const std::exception& err)
            {
                throw err;
            }
        }

        {
            auto screen_config = j.at("screen").at(std::to_string(world_rank));
            screen_offset_x = screen_config.at("offsetX");
            screen_offset_y = screen_config.at("offsetY");
            resource_path = screen_config.at("resourcePath");
            std::cout << "node " << world_rank << " resource path: " << resource_path << std::endl;
        }
    }
    ~WindowManager(){
        MPI_Finalize();
    }
    int GetWindowRank() const{
        return world_rank;
    }
    int GetWindowNum() const{
        return world_size;
    }
    int GetWindowRowNum() const{
        return row_num;
    }
    int GetWindowColNum() const{
        return col_num;
    }
    int GetNodeWindowWidth() const{
        return world_window_width/col_num;
    }
    int GetNodeWindowHeight() const{
        return world_window_height/row_num;
    }
    int GetWorldWindowWidth() const{
        return world_window_width;
    }
    int GetWorldWindowHeight() const{
        return world_window_height;
    }
    int GetWorldRankOffsetX() const{
        return world_rank % col_num;
    }
    int GetWorldRankOffsetY() const{
        return world_rank / col_num;
    }
    void GetWorldVolumeSpace(float& x,float& y,float& z) const{
        x=space_x;
        y=space_y;
        z=space_z;
    }
    int GetNodeScreenOffsetX() const{
        return screen_offset_x;
    }
    int GetNodeScreenOffsetY() const{
        return screen_offset_y;
    }
    std::string GetNodeResourcePath() const{
        return resource_path;
    }
    int GetFrameTimeLock() const{
        return frame_time_lock;
    }
    std::string GetRendererBackend() const{
        return renderer_backend;
    };
    int GetGPUIndex() const{
        return iGPU;
    }
    bool IsRoot(int rank) const{
        return rank == root_rank;
    }
    bool IsRoot() const{
        return world_rank == root_rank;
    }
    const auto& GetTFMap() const{
        return tf_map;
    }
    const auto& GetAdvanceOptions() const{
        return advance_options;
    }
public:
    //same for each node
    int root_rank = 0;
    int world_window_width;
    int world_window_height;
    int world_size;
    int row_num;
    int col_num;
    float space_x,space_y,space_z;
    int frame_time_lock;
    int iGPU;
    std::string renderer_backend;

    AdvanceOptions advance_options;

    std::map<uint8_t,std::array<double,4>> tf_map;

    char processName[MPI_MAX_PROCESSOR_NAME];
    int processNameLength;
    int world_rank;
    int screen_offset_x;
    int screen_offset_y;
    std::string resource_path;
};


