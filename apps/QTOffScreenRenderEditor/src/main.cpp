//
// Created by csh on 10/20/2021.
//
#include"QTOffScreenRenderEditor.h"
#include <QtWidgets>
#include <QApplication>

int main(int argc,char** argv)
{
    QApplication app(argc,argv);

    QTOffScreenRenderEditor w;
    w.show();

    return app.exec();
}
