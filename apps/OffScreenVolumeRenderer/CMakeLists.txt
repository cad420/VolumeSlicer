set(FFmpeg_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ffmpeg/include)
set(FFmpeg_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ffmpeg/bin)
set(FFmpeg_DLLS
        ${FFmpeg_LIBRARY_DIR}/avcodec-59.dll
        ${FFmpeg_LIBRARY_DIR}/avdevice-59.dll
        ${FFmpeg_LIBRARY_DIR}/avfilter-8.dll
        ${FFmpeg_LIBRARY_DIR}/avformat-59.dll
        ${FFmpeg_LIBRARY_DIR}/avutil-57.dll
        ${FFmpeg_LIBRARY_DIR}/swresample-4.dll
        ${FFmpeg_LIBRARY_DIR}/swscale-6.dll)
file(COPY ${FFmpeg_DLLS}
        DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

list(
        APPEND
        OffScreenVolumeRenderer_SRCS
        src/main.cpp
)
add_executable(OffScreenVolumeRenderer ${OffScreenVolumeRenderer_SRCS})
target_link_directories(
        OffScreenVolumeRenderer
        PRIVATE
        ${FFmpeg_LIBRARY_DIR}
)
target_include_directories(
        OffScreenVolumeRenderer
        PRIVATE
        ${FFmpeg_INCLUDE_DIR}
)
target_link_libraries(
        OffScreenVolumeRenderer
        PRIVATE
        VS_CORE
)