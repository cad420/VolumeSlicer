
option(VS_BUILD_GUI_OFF_RENDERER "build" ON)
option(VS_BUILD_OFF_RENDERER_TEST "build" ON)

list(
        APPEND
        OffScreenVolumeRenderer_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/VideoCapture.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/OffScreenVolumeRenderer.cpp
)
add_executable(OffScreenVolumeRenderer src/main.cpp ${OffScreenVolumeRenderer_SRCS})

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv)
find_package(OpenCV REQUIRED)

target_include_directories(
        OffScreenVolumeRenderer
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
        OffScreenVolumeRenderer
        PRIVATE
        VS_CORE
        ${OpenCV_LIBS}
)

target_compile_features(
        OffScreenVolumeRenderer
        PRIVATE
        cxx_std_17
)

message(status ${OpenCV_DLLS})
set(OpenCV_DLLS
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/x64/vc15/bin/opencv_world453.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/x64/vc15/bin/opencv_world453d.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/x64/vc15/bin/opencv_videoio_ffmpeg453_64.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/x64/vc15/bin/opencv_videoio_msmf453_64.dll
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/x64/vc15/bin/opencv_videoio_msmf453_64d.dll
        )
file(COPY ${OpenCV_DLLS}
        DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

if(VS_BUILD_GUI_OFF_RENDERER)
    add_subdirectory(gui)
endif()

if(VS_BUILD_OFF_RENDERER_TEST)
    add_subdirectory(test)
endif()