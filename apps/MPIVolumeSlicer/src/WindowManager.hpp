//
// Created by wyz on 2021/7/12.
//

#pragma once
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

        try
        {
            world_window_width = j.at("width");
            world_window_height = j.at("height");
            row_num = j.at("row");
            col_num = j.at("col");
            auto space = j.at("space");
            space_x = space.at(0);
            space_y = space.at(1);
            space_z = space.at(2);

            auto screen_config = j.at("screen").at(std::to_string(world_rank));
            screen_offset_x = screen_config.at("offsetX");
            screen_offset_y = screen_config.at("offsetY");
            resource_path = screen_config.at("resourcePath");
            std::cout << "node " << world_rank << " resource path: " << resource_path << std::endl;

            auto raw_config = j.at("raw");
            raw_resource_path = raw_config.at("path");
            raw_lod = raw_config.at("lod");
            raw_dim_x = raw_config.at("dim")[0];
            raw_dim_y = raw_config.at("dim")[1];
            raw_dim_z = raw_config.at("dim")[2];
        }
        catch (std::exception const& err)
        {
            std::cout<<"WindowManager open file cause error: "<<err.what()<<std::endl;
            exit(-1);
        }
        catch (...)
        {
            std::cout<<"WindowManager open file failed"<<std::endl;
            exit(-1);
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
    bool IsRoot(int rank) const{
        return rank == root_rank;
    }
    bool IsRoot() const{
        return world_rank == root_rank;
    }
    std::string GetRawResourcePath(int& lod) const{
        lod=raw_lod;
        return raw_resource_path;
    }
    void GetRawDim(int& dim_x,int& dim_y,int& dim_z) const{
        dim_x=raw_dim_x;
        dim_y=raw_dim_y;
        dim_z=raw_dim_z;
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

    char processName[MPI_MAX_PROCESSOR_NAME];
    int processNameLength;
    int world_rank;
    int screen_offset_x;
    int screen_offset_y;
    std::string resource_path;

    std::string raw_resource_path;
    int raw_lod;
    int raw_dim_x,raw_dim_y,raw_dim_z;
};


