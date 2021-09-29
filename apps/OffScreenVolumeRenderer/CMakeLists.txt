set(FFmpeg_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ffmpeg/include)
set(FFmpeg_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ffmpeg/bin)
set(FFmpeg_LIBS
        avcodec
        avdevice
        avfilter
        avformat
        avutil
        swresample
        swscale)
set(FFmpeg_DLLS
        ${FFmpeg_LIBRARY_DIR}/avcodec-59.dll
        ${FFmpeg_LIBRARY_DIR}/avdevice-59.dll
        ${FFmpeg_LIBRARY_DIR}/avfilter-8.dll
        ${FFmpeg_LIBRARY_DIR}/avformat-59.dll
        ${FFmpeg_LIBRARY_DIR}/avutil-57.dll
        ${FFmpeg_LIBRARY_DIR}/swresample-4.dll
        ${FFmpeg_LIBRARY_DIR}/swscale-6.dll
        )
file(COPY ${FFmpeg_DLLS}
        DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

list(
        APPEND
        OffScreenVolumeRenderer_SRCS
        src/main.cpp
        src/VideoCapture.cpp
        src/OffScreenVolumeRenderer.cpp
)
add_executable(OffScreenVolumeRenderer ${OffScreenVolumeRenderer_SRCS})

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv)
find_package(OpenCV REQUIRED)
target_link_directories(
        OffScreenVolumeRenderer
        PRIVATE

)
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

add_subdirectory(test)