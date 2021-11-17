//
// Created by wyz on 2021/10/26.
//
#include "SlicerServerApplication.hpp"
using namespace vs::remote;
int main(int argc,char** argv){
    SlicerServerApplication app{};
    return app.run(argc,argv);
}