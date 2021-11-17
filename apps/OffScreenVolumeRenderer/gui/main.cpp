//
// Created by wyz on 2021/11/15.
//
#include "OffScreenVolumeRenderWindow.hpp"
#include <QApplication>
int main(int argc,char** argv){
    QApplication app(argc,argv);
    if(argc>1){
        OffScreenVolumeRenderWindow w;
        w.show();
        w.open(argv[1]);
        w.StartRender();
        return QApplication::exec();
    }
    else{
        OffScreenVolumeRenderWindow w;
        w.show();
        return QApplication::exec();
    }


}