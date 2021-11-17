//
// Created by wyz on 2021/11/15.
//
#include "OffScreenVolumeRenderWindow.hpp"
#include <QApplication>
int main(int argc,char** argv){
    QApplication app(argc,argv);
    if(argc>1){
        OffScreenVolumeRenderWindow w(argv[1]);
        w.show();
        return QApplication::exec();
    }
    else{
        OffScreenVolumeRenderWindow w;
        w.show();
        return QApplication::exec();
    }


}