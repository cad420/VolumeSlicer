//
// Created by wyz on 2021/7/12.
//

#ifndef VOLUMESLICER_WINDOWMANAGER_HPP
#define VOLUMESLICER_WINDOWMANAGER_HPP
#include <json.hpp>
#include <fstream>
#include <mpi.h>
#include <iostream>
using nlohmann::json;
class WindowManager{
public:
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
        auto space=j.at("space");
        space_x=space.at(0);
        space_y=space.at(1);
        space_z=space.at(2);

        auto screen_config=j.at("screen").at(std::to_string(world_rank));
        screen_offset_x=screen_config.at("offsetX");
        screen_offset_y=screen_config.at("offsetY");
        resource_path=screen_config.at("resourcePath");
        std::cout<<"node "<<world_rank<<" resource path: "<<resource_path<<std::endl;
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
public:
    //same for each node
    int world_window_width;
    int world_window_height;
    int world_size;
    int row_num;
    int col_num;
    float space_x,space_y,space_z;

    char processName[MPI_MAX_PROCESSOR_NAME];
    int processNameLength;
    int world_rank;
    int screen_offset_x;
    int screen_offset_y;
    std::string resource_path;
};

#endif //VOLUMESLICER_WINDOWMANAGER_HPP
