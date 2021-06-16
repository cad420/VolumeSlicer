cmake_minimum_required(VERSION 3.17)

project(VolumeSlicer CXX C CUDA)

include_directories(include)

add_subdirectory(src)

add_subdirectory(test)

add_subdirectory(apps/QTVolumeSlicer)