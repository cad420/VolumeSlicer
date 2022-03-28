include(${PROJECT_SOURCE_DIR}/third_party/VolumeCompression.cmake)
list(
        APPEND
        BLOCK_VOLUME_PROVIDER_SRCS
        block_volume_provider_plugin.cpp
        volume_transform.cu
)
add_library(BlockVolumeProviderPlugin
        SHARED
        ${BLOCK_VOLUME_PROVIDER_SRCS})
file(
        GLOB FFmpeg_LIBS
        "${PROJECT_SOURCE_DIR}/third_party/ffmpeg-master-latest-win64-gpl-shared/lib/*.lib")
set(FFmpeg_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/ffmpeg-master-latest-win64-gpl-shared/include)
message(STATUS ${FFmpeg_LIBS})

target_link_libraries(
        BlockVolumeProviderPlugin
        PRIVATE
        VS_CORE
        voxel_uncompress
        ${FFmpeg_LIBS}
)
target_include_directories(
        BlockVolumeProviderPlugin
        PRIVATE
        ${FFmpeg_INCLUDE_DIR}
)
set_target_properties(
        BlockVolumeProviderPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins
)