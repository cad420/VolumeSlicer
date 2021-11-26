cmake_minimum_required(VERSION 3.17)

project(VolumeSlicer CXX C CUDA)

option(VS_BUILD_TEST "" ON)
option(VS_BUILD_TOOLS "" ON)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

add_subdirectory(src)

add_subdirectory(test)

add_subdirectory(apps)

add_subdirectory(plugin)

add_subdirectory(tools)