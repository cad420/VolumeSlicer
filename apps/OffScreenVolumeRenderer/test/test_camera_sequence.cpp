//
// Created by wyz on 2021/9/29.
//
#include <fstream>
#include <json.hpp>
int main(){
    nlohmann::json j;
    j["frame_count"] = 30*60;
    j["property"] = {"zoom","pos","look_at","up","right"};
    for(int i=0;i<30*60;i++){
        std::string idx="frame_"+std::to_string(i+1);
        j[idx]={
            12,
            {2.5,6.5,8.0 - i * 0.0024},
            {2.5,6.5,0.0},
            {0.0,1.0,0.0},
            {1.0,0.0,0.0}
        };
    }
    std::ofstream out("camera_sequence_config.json");
    out << j <<std::endl;
    out.close();
}