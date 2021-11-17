set(CMAKE_PREFIX_PATH C:\\Qt\\6.1.0\\msvc2019_64\\lib\\cmake)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
find_package(OpenMP REQUIRED)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

file(
        GLOB
        OffScreenVolumeRenderGUI_SRC
        "*.cpp"
        "*.c"
        "*.hpp"
        "*.h"
)

add_executable(OffScreenVolumeRenderGUI
        ${OffScreenVolumeRenderGUI_SRC}
        ../src/OffScreenVolumeRenderer.cpp
        ../src/VideoCapture.cpp
        )
target_include_directories(
        OffScreenVolumeRenderGUI
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ../src
)
target_link_libraries(
        OffScreenVolumeRenderGUI
        PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        ${OpenCV_LIBS}
        VS_CORE
)
embed_shaders(
        CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/shaders.hpp.in
        TARGET ${CMAKE_CURRENT_SOURCE_DIR}/shaders.hpp
        DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders
)
