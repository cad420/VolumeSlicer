//
// Created by wyz on 2021/10/18.
//
#include "VideoCapture.hpp"
#include <VolumeSlicer/Render/render.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
int main(){
    ::VideoCapture vc("test_img_file_tovideo.avi",1200,900,30);
    for(int i=0;i<10;i++){
        std::string img_name="images/result333_frame_"+std::to_string(i)+".jpeg";
        auto img = imread(img_name);

        vc.AddFrame(img.data);
    }

    return 0;
}