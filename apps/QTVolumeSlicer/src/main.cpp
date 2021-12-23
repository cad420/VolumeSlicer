//
// Created by wyz on 2021/6/11.
//
#include "QTVolumeSlicerApp.hpp"
#include <QtWidgets>
#include <QApplication>

#ifdef NDEBUG
#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#endif

int main(int argc,char** argv)
{
    QApplication app(argc,argv);

    VolumeSlicerMainWindow w;
    w.show();

    return app.exec();
}
