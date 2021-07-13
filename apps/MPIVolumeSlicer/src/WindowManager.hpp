//
// Created by wyz on 2021/7/12.
//

#ifndef VOLUMESLICER_WINDOWMANAGER_HPP
#define VOLUMESLICER_WINDOWMANAGER_HPP

class WindowManager{
public:
    WindowManager(const char* config_file){

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
        return world_rank / row_num;
    }
public:
    //same for each node
    int world_window_width;
    int world_window_height;
    int world_size;
    int row_num;
    int col_num;

    int world_rank;
    int screen_offset_x;
    int screen_offset_y;
    std::string resource_path;
};

#endif //VOLUMESLICER_WINDOWMANAGER_HPP
