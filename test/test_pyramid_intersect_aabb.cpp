//
// Created by wyz on 2021/8/23.
//

#include <Common/boundingbox.hpp>
using namespace vs;
int main(){

    Pyramid pyramid{
            {0.f,0.f,0.f},
            {3.f,2.f,6.f},
            {-3.f,2.f,6.f},
            {3.f,-2.f,6.f},
            {-3.f,-2.f,6.f}
    };
    AABB aabb(
            {-0.1f,-0.1f,5.f},
            {0.1f,0.1f,5.2f}
            );
    spdlog::info("test 0: {0}.",pyramid.intersect_aabb(aabb)?"true":"false");
    aabb=AABB(
            {-0.1f,-0.1f,-5.3f},
            {0.1f,0.1f,-5.1f}
    );
    spdlog::info("test 1: {0}.",pyramid.intersect_aabb(aabb)?"true":"false");

    aabb=AABB(
            {-0.1f,-0.1f,-0.3f},
            {0.1f,0.1f,-0.1f}
    );
    spdlog::info("test 2: {0}.",pyramid.intersect_aabb(aabb)?"true":"false");
    aabb=AABB(
            {-0.1f,-0.1f,-0.1f},
            {0.1f,0.1f,0.1f}
    );
    spdlog::info("test 3: {0}.",pyramid.intersect_aabb(aabb)?"true":"false");
    return 0;
}