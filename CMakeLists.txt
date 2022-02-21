cmake_minimum_required(VERSION 3.16)
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

project(VolumeSlicer CXX C CUDA)
#find_package(CUDA REQUIRED) # ???

option(VS_BUILD_TEST "" OFF)
option(VS_BUILD_TOOLS "" OFF)
option(VS_BUILD_PLUGINS "" OFF)
option(VS_INSTALL "" ON)
option(VS_BUILD_OPENGL_RENDERER "" OFF)


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../lib")

include_directories(include)

add_subdirectory(src)

if(VS_BUILD_TEST)
    add_subdirectory(test)
endif()

add_subdirectory(apps)

if(VS_BUILD_PLUGINS)
    add_subdirectory(plugin)
endif()

if(VS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()